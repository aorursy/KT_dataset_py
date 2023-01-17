# useful when you move it to your machine:
path_stem = '../input/'

verbose = True
use_agg = False # turn on when run without a graphics head

MINIMAL_BLURRING_KERNEL_SIZE = 4.
# ----------------------------------------------------------------------------
# IMPORTING

# import train data
def import_data (folder='stage1_train'):
    print('Importing train data...')
    from tqdm import tqdm
    import os

    DATA = {}; DATA['images'] = {};# DATA['masks'] = {}
    for dataset in tqdm(os.listdir(path_stem+folder)):
        for typ in ['images']: #, 'masks']:
            DATA[typ][dataset] = os.listdir(path_stem+folder+'/'+dataset+'/'+typ)
        # report if anything unusual found
        if len(DATA['images'][dataset]) != 1:
            print('   %i images for %s' % (len(DATA['images']),dataset))
        #if len(DATA['masks'][dataset]) < 1:
        #    print('   no masks for %s' % dataset)
    print('done. Imported %i datasets' % len(DATA['images']))

    return DATA

# image normalization functions
def normalize_color (img):
    import numpy as np
    maxs = np.max(img, axis=(0,1))
    img = np.array(img) / max(maxs)
    return img
def normalize_gray (img):
    import numpy as np
    maxx = np.max(img)
    img = np.array(img) / maxx
    return img
# ----------------------------------------------------------------------------
# BACKGROUND EXTRACTION

def detect_background (dataset, imgname, visualize=False, folder='stage1_train', n_channels=128):

    import matplotlib
    if use_agg:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    from scipy.optimize import curve_fit, bisect

    # definitions need to be copied here for multithreading to work
    def normalize_gray (img):
        import numpy as np
        maxx = np.max(img)
        img = np.array(img) / maxx
        return img
    def gaussian (x, A, x0, log_sigma):
        sigma = np.exp(log_sigma)
        return A * np.exp(-(x-x0)**2/(2.0*sigma**2))
    def lorentzian (x, A, x0, log_sigma):
        sigma = np.exp(log_sigma)
        return A * sigma / ( (x-x0)**2 + sigma**2)

    try:
        if verbose:
            print('Reading the image... ', end='', flush=True)
        imgpath = path_stem+folder+'/'+dataset+'/images/'+imgname
        img = cv2.imread(imgpath, 0)
        img = normalize_gray(img)
        if len(set(np.array(img).flatten())) == 1:
            print(' Image empty, moving on.')
            return -1, -1
        if verbose:
            print('done.', flush=True)
        img0 = 1. * img
        img_range = np.min(img), np.max(img)
        n_bins = max(int(len(set(np.array(img).flatten()))/3.), 10)
        force_gmm = False
        force_gmm_brightest = False
        blurring_kernel_size = MINIMAL_BLURRING_KERNEL_SIZE

        for trial_idx in range(10): # redo if noisy image detected, see below

            try:

                if visualize:
                    plt.clf()
                    plt.imshow(img)
                    plt.title('Current image state')
                    plt.show()
                    plt.clf()
                vals_sparse, _, _ = plt.hist(np.array(img).flatten(), 32)
                plt.clf()
                vals, bins, _ = plt.hist(np.array(img).flatten(), n_bins)
                bins = np.array(bins); vals = np.array(vals)
                bins = 0.5*(np.roll(bins, -1)[:-1] + bins[:-1])
                if visualize:
                    plt.title('Pixel brightness histogram. Black vertical solid line: threshold.')
                    plt.plot(bins, vals, linewidth=2, color='red')
                    plt.xlim(0.,1.)

                # manual fitting of a lorentzian (bg) and a gaussian (cells) performs better if the histogram is background-dominated:
                if max(vals_sparse) > 0.35 * np.sum(vals_sparse) and not force_gmm:
                    # fit the main maximum (usually: background)
                    if verbose:
                        print('Finding the maximum... ', end='', flush=True)
                    posmax = bins[np.where(vals == max(vals))[0][0]]
                    first_gauss, cov = curve_fit(lambda x, A, log_sigma : lorentzian(x, A, posmax, log_sigma), bins, vals, p0=[max(vals), np.log(0.1)], bounds=([0.,-np.inf], [np.inf, 0.]))
                    A, log_sigma = first_gauss
                    if verbose:
                        print('done.', flush=True)
                    #correct for position
                    if verbose:
                        print('Correcting for position... ', end='', flush=True)
                    first_gauss, cov = curve_fit(lambda x, x0 : lorentzian(x, A, x0, log_sigma), bins, vals, p0=[posmax], bounds=img_range)
                    posmax = first_gauss[0]
                    first_gauss, cov = curve_fit(lambda x, A, log_sigma : lorentzian(x, A, posmax, log_sigma), bins, vals, p0=[max(vals), np.log(0.1)], bounds=([0.,-np.inf], [np.inf, np.inf]))
                    A, log_sigma = first_gauss
                    if verbose:
                        print('done.', flush=True)
                    #plot
                    if visualize:
                        plt.axvline(posmax, color='g', linewidth=2)
                        plt.plot(bins, lorentzian(bins, A, posmax, log_sigma), color='g', linewidth=2)
                    # save the fit
                    first_gauss = [A, posmax, log_sigma]

                    # fit the secondary maximum (usually: cells)
                    if verbose:
                        print('Fitting the secondary maximum... ', end='', flush=True)
                    primary_vals = lorentzian(bins, A, posmax, log_sigma)
                    primary_max_idx = np.argmax(primary_vals)
                    sigma_idx = max(int(len(primary_vals) * np.exp(log_sigma)), 2)
                    mask = max(0, primary_max_idx-2*sigma_idx), min(len(bins), primary_max_idx+2*sigma_idx)
                    vals = vals / (1. + primary_vals)
                    vals[mask[0]:mask[1]] = 0. #0.5*(vals[mask[0]]+vals[mask[1]])
                    if visualize:
                        plt.plot(bins, vals, color='k', linewidth=2)
                    posmax_secondary = bins[np.argmax(vals)]
                    print(posmax_secondary)
                    if abs(posmax - posmax_secondary) < 0.1:
                        if posmax > 0.5:
                            posmax_secondary = 0.5 * posmax
                        else:
                            posmax_secondary = 1. - 0.5*(1.-posmax)
                    print("max vals", max(vals))
                    second_gauss, cov = curve_fit(lambda x, A, log_sigma : gaussian(x, A, posmax_secondary, log_sigma), bins, vals, p0=[max(vals), np.log(0.4)], bounds=([5.,-np.inf], [max(vals), np.log(1.)]))
                    A, log_sigma = second_gauss
                    if verbose:
                        print('done.', flush=True)
                    #correct for position
                    if verbose:
                        print('Correcting for position... ', end='', flush=True)
                    second_gauss, cov = curve_fit(lambda x, x0 : gaussian(x, A, x0, log_sigma), bins, vals, p0=[posmax_secondary], bounds=img_range)
                    posmax_secondary = second_gauss[0]
                    second_gauss, cov = curve_fit(lambda x, A, log_sigma : gaussian(x, A, posmax_secondary, log_sigma), bins, vals, p0=[max(vals), np.log(0.1)], bounds=([5.,-np.inf], [max(vals), np.log(1.)]))
                    A, log_sigma = second_gauss
                    if verbose:
                        print('done.', flush=True)
                    #plot
                    if visualize:
                        plt.axvline(posmax_secondary, color='orange', linewidth=2)
                        plt.plot(bins, gaussian(bins, A, posmax_secondary, log_sigma), color='orange', linewidth=2)
                    # save the fit
                    second_gauss = [A, posmax_secondary, log_sigma]

                    # find where the two fits are equal
                    if verbose:
                        print('Splitting... ', end='', flush=True)
                    try:
                        splitter = bisect(lambda x : lorentzian(x, *first_gauss) - gaussian(x, *second_gauss), first_gauss[1], second_gauss[1])
                    except Exception as e:
                        splitter = 0.5 * (first_gauss[1] + second_gauss[1])
                        print('Bisection failed, defaulting to average of maxima.')
                    if verbose:
                        print('done.', flush=True)
                    if visualize:
                        plt.axvline(splitter, color='k', linewidth=2)
                        plt.show()
                else: # use a Gaussian mixture model if cells contribute significantly to the histogram
                    from sklearn.mixture import GaussianMixture
                    X = np.array(img).flatten().reshape((-1,1)) # samples
                    models = [GaussianMixture(n).fit(X) for n in range(2,5)]
                    AIC = [m.aic(X) for m in models]
                    model = models[np.argmin(AIC)] # best model
                    x_axis = np.linspace(0.,1.,32)
                    p = model.predict_proba(x_axis.reshape((-1,1))).transpose()
                    if visualize:
                        for y in p:
                            plt.plot(x_axis, y)
                    splitters = []
                    for y in p:
                        pos = np.where(y > 0.5)[0]
                        if len(pos) > 0:
                            splitters.append(x_axis[max(np.where(y > 0.5)[0])])
                        else:
                            splitters.append(x_axis[np.argmax(y)])
                    splitters = np.array(sorted(splitters))[:-1]
                    if visualize:
                        for splitter in splitters:
                            plt.axvline(splitter)
                    if not force_gmm_brightest:
                        splitter = splitters[0]
                    else:
                        tail = 1
                        splitter = splitters[-tail]
                        while len(np.where(img0 > splitter)[0]) < 0.05 * np.product(img0.shape):
                            tail += 1
                            if tail == len(splitters):
                                break
                            splitter = splitters[-tail]


                if verbose:
                    print("Threshold: ", splitter)

                if visualize:
                    plt.show()
                    plt.clf()
                    plt.subplot(121)
                    img_to_show = 1. * img
                    mask = np.where(img > splitter)
                    img_to_show[mask] = 0.
                    plt.imshow(img_to_show)
                    plt.subplot(122)
                    img_to_show = 1. * img
                    mask = np.where(img < splitter)
                    img_to_show[mask] = 0.
                    plt.imshow(img_to_show)
                    plt.suptitle('Background and foreground (yet unclassified)')
                    plt.show()

                # Perform statistics of the contiguous regions in the flattened image thresholded by splitter. If the sizes are only few pixels, the image is likely noise-dominated -- convolve it with a gaussian kernel and redo the analysis. Note: I added this feature to the code after viewing the cases that failed during 1st pass of stage2_test_final.
                flattened_cells_image = np.array(1. * img)
                mask = np.where(img < splitter)
                flattened_cells_image[mask] = 0
                mask = np.where(img > splitter)
                flattened_cells_image[mask] = 1
                flattened_cells_image = flattened_cells_image.astype(np.int)
                flattened_cells_image = np.array(flattened_cells_image).flatten()
                if np.sum(flattened_cells_image) > 0.85 * len(flattened_cells_image) and not force_gmm:
                    if verbose:
                        print("Threshold too low, forcing GMM.")
                    force_gmm = True
                    continue
                flattened_cells_image = np.array([flattened_cells_image[:-1], flattened_cells_image[1:]]).transpose()
                contiguous_starts = list(np.where(list(map(lambda x : x[0] == 0 and x[1] == 1, flattened_cells_image)))[0])
                contiguous_stops = list(np.where(list(map(lambda x : x[0] == 1 and x[1] == 0, flattened_cells_image)))[0])
                if flattened_cells_image[0][0] > 0: contiguous_starts.insert(0,-1)
                if flattened_cells_image[-1][1] > 0: contiguous_stops.append(len(flattened_cells_image))
                contiguous_sizes = np.array(contiguous_stops) - np.array(contiguous_starts)
                if (np.median(contiguous_sizes)) < 3.:
                    if len(contiguous_sizes) < 10 and force_gmm == False:
                        if verbose:
                            print("Simple background detection failed. Forcing Gaussian Mixture Model.")
                        force_gmm = True
                    else:
                        if verbose:
                            print("Noisy image detected. Restarting background detection with a blurred image.")
                        # convolve with a flat kernel and redo
                        blurring_kernel_size = int(1.5 * blurring_kernel_size)
                        img = cv2.filter2D(img0, -1, np.ones((blurring_kernel_size,blurring_kernel_size), np.float32)/(blurring_kernel_size**2))
                        #img = cv2.GaussianBlur(img, (5,5), 0)
                        #n_bins = 32
                        if not force_gmm:
                            force_gmm_brightest = True # in noise dominated images, the nuclei seem to be the brightest objects (imaging method?)
                            force_gmm = True # force the Gaussian mixture model background (two-element model is too simple for blurred images)
                else:
                    break

            except Exception as e:
                print("Exception caught: %s. " % e, end='')
                if n_bins < 129:
                    print('Attempting to use more bins.')
                    n_bins = n_bins * 2
                else:
                    print('Forcing GMM.')
                    n_bins = 32
                    force_gmm = True
                continue

        return splitter, blurring_kernel_size
        
    except Exception as e:
        print("Could not process image: %s. Error: %s" % (imgname, e))
        return 0.5
def detect_all_backgrounds (DATA, dataset, folder='stage1_train', visualize=False):
    print('Detecting backgrounds for %s' % dataset)
    thresholds = {}
    for imgname in DATA['images'][dataset]:
        thresholds[imgname] = detect_background(dataset, imgname, folder=folder, visualize=visualize)
    return thresholds
folder = 'stage2_test_final' # directory holding the dataset to be analysed

# import data
DATA = import_data(folder=folder)
datasets = list(DATA['images'].keys())
# case 1: pixel brightness histogram fit as a Lorentzian and a Gaussian
dataset = '0154f82b5f214a91e3475d0b14d9bbe93960b7b1f925da9bb8e2b4aa65ec006a'
thresholds = detect_all_backgrounds(DATA, dataset, folder=folder, visualize=True)
# case 2: using blurring to fix noisy images
dataset = '2ac1e92c749e54b5bad111be1220d0ea07dd176a48c51fac4c212f2b7aeec8af'
splitters = detect_all_backgrounds(DATA, dataset, folder=folder, visualize=True)
# case 4: using a Gaussian Mixture Model decomposition
dataset = '63bc2a0c564882b414349b3485208e50b47aa56f8f0e9cc1fe1e0fe8cf60d1f6'
thresholds = detect_all_backgrounds(DATA, dataset, folder=folder, visualize=True)
# case 5: it's still far from perfect
dataset = 'ec54bb85006c1a4ebfcc87aa0d369c6ce1bf85ef898c613b490c9dcce379e096'
thresholds = detect_all_backgrounds(DATA, dataset, folder=folder, visualize=True)
