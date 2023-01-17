import numpy as np

import matplotlib.image as mpimg

import matplotlib.pyplot as plt
orig_img = np.array(mpimg.imread('../input/origimg.jpg')).astype(int)

orig_img[orig_img<128] = -1

orig_img[orig_img>=128] = 1



plt.figure()

plt.axis('off')

plt.imshow(orig_img, cmap='binary_r')



noisy_img = np.array(mpimg.imread('../input/noisyimg.jpg')).astype(int)

noisy_img[noisy_img<128] = -1

noisy_img[noisy_img>=128] = 1



plt.figure()

plt.axis('off')

plt.imshow(noisy_img, cmap='binary_r')

print ('Image:')

# compute the error rate between the noisy image and the original image

def compute_error_rate(img1, img2):

    err = abs(img1 - img2) / 2

    return np.sum(err) / np.size(img2) * 100



print ('Percentage of mismatched pixels in noisy image: %.6f%%' % compute_error_rate(orig_img, noisy_img))
def compute_prob(X, Y, i, j, m, n, beta, eta, x_value):

    

    result = beta * Y[i][j] * x_value

    

    if i > 0:

        result += eta * x_value * X[i-1][j]

    if i < m-1:

        result += eta * x_value * X[i+1][j]

    if j > 0:

        result += eta * x_value * X[i][j-1]

    if j < n-1:

        result += eta * x_value * X[i][j+1]

    

    

    return result
def denoise_image(Y, orig, beta, eta):

    m, n = np.shape(Y)

    X = np.copy(Y)

    max_iter = 5

    

    for k in range(max_iter):

        for i in range(m):

            for j in range(n):

                

                p_pos = compute_prob(X, Y, i, j, m, n, beta, eta, 1)

                p_neg = compute_prob(X, Y, i, j, m, n, beta, eta, -1)

                

                if p_pos > p_neg:

                    X[i][j] = 1

                else:

                    X[i][j] = -1

                    

        print('Iteration number:', k+1)

        print ('Percentage of mismatched pixels: %.6f%%' % compute_error_rate(orig, X))

    return X
beta = 1.0

eta = 2.1

denoised_img = denoise_image(noisy_img, orig_img, beta, eta)



plt.figure()

plt.axis('off')

plt.imshow(denoised_img, cmap='binary_r')

print ('Percentage of mismatched pixels in denoised image: %.6f%%' % compute_error_rate(orig_img, denoised_img))

print ('Denoised Image:')