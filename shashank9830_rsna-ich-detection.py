import matplotlib.pyplot as plt

import pydicom

import random
%matplotlib inline
def get_window_image(ds, window='default', rescale=True):

    ''' Return image for particular window_type'''



    if window == 'default':

        if type(ds.WindowCenter) == pydicom.multival.MultiValue:

            window_center = ds.WindowCenter[0]

        else:

            window_center = ds.WindowCenter



        if type(ds.WindowWidth) == pydicom.multival.MultiValue:

            window_width = ds.WindowWidth[0]

        else:

            window_width = ds.WindowWidth

    elif window == 'brain':

        window_center, window_width = 40, 80

    elif window == 'subdural-min':

        window_center, window_width = 50, 130

    elif window == 'subdural-mid':

        window_center, window_width = 75, 215

    elif window == 'subdural-max':

        window_center, window_width = 100, 300

    elif window == 'tissue-min':

        window_center, window_width = 20, 350

    elif window == 'tissue-mid':

        window_center, window_width = 40, 375

    elif window == 'tissue-max':

        window_center, window_width = 60, 400

    elif window == 'bone':

        window_center, window_width = 600, 2800

    elif window == 'grey_white':

        window_center, window_width = 32, 8

    else:

        raise ValueError(f'{window} is not a valid window.')



    img = ds.pixel_array*ds.RescaleSlope + ds.RescaleIntercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img[img < img_min] = img_min

    img[img > img_max] = img_max



    if rescale:

        img = (img - img_min) / (img_max - img_min)



    return img
def show_image(images, file_ID):

    ''' Plot the image using imshow '''



    fig, ax = plt.subplots(5, 2, figsize = (10, 24), gridspec_kw = {'hspace': 0.2, 'wspace': 0.1})



    titles = [['Default Window', 'Brain Window'],

                ['Subdural Window (Min)', 'Subdural Window (Max)'],

                ['Subdural Window (Mid)', 'Tissue Window (Mid)'],

                ['Tissue Window (Min)', 'Tissue Window (Max)'],

                ['Bone Window', 'Grey-White Window']]

    

    fig.suptitle(f'"ID_{file_ID}.dcm" multi-window view', fontsize='14')



    i = 0

    for x in range(5):

        for y in range(2):

            ax[x, y].set_title(titles[x][y])

            ax[x, y].imshow(images[i], cmap=plt.cm.bone)

            i += 1



    plt.subplots_adjust(top=0.95)

    plt.show()
def display_image_in_diff_windows(file_ID):



    stage_2_dir = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train'

    filename = stage_2_dir + '/ID_' + file_ID + '.dcm'

    dicom_data = pydicom.dcmread(filename)



    windows = ['default', 'brain', 'subdural-min', 'subdural-max', 'subdural-mid',

                'tissue-mid', 'tissue-min', 'tissue-max', 'bone', 'grey_white']

    images = []



    for window in windows:

        img = get_window_image(dicom_data, window)

        images.append(img)



    show_image(images, file_ID)
interesting_cases = ['00c4d3226', '00ad28f0c', '00c7b2578', '00c2f083a']
display_image_in_diff_windows(interesting_cases[0])
display_image_in_diff_windows(interesting_cases[1])
display_image_in_diff_windows(interesting_cases[2])
display_image_in_diff_windows(interesting_cases[3])