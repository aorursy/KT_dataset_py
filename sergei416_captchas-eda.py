import cv2

import os

import matplotlib.pyplot as plt

import numpy as np

import random
tot_size = len(os.listdir('/kaggle/input/captcha-images/'))

num = random.randint(0, tot_size)

print(num)

alli = list(os.listdir('/kaggle/input/captcha-images/'))

file = alli[num]

print(file)
solution = file.split('.')[0]

solution
hi = cv2.imread('/kaggle/input/captcha-images/' + file)
plt.imshow(hi, cmap="gray")

plt.axis('off')

plt.show()
# convert to RGB

image = cv2.cvtColor(hi, cv2.COLOR_BGR2RGB)

# convert to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# create a binary thresholded image

_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

# show it

plt.imshow(binary, cmap="gray")

plt.axis('off')

plt.show()
# find the contours from the thresholded image

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw all contours

image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
# show the image with the drawn contours

plt.imshow(image)

plt.axis('off')

plt.show()
x = {}

for m in range(len(contours)):

    mini = 1000

    for k in range(len(contours[m])):

        first = contours[m][k][0][0]

        if first < mini:

            mini = first

    mini1 = mini

    #print(mini1)

    

    mini = 1000

    for k in range(len(contours[m])):

        first = contours[m][k][0][1]

        if first < mini:

            mini = first

    mini2 = mini

    #print(mini2)

    

    maxi = 0

    for k in range(len(contours[m])):

        first = contours[m][k][0][1]

        if first > maxi:

            maxi = first

    maxi2 = maxi

    #print(maxi2)

    

    maxi = 0

    for k in range(len(contours[m])):

        first = contours[m][k][0][0]

        if first > maxi:

            maxi = first

    maxi1 = maxi

    #print(maxi1)

    x[m] = maxi2 - mini2
biggie = sorted(x, key=x.get)
s = {}

def plotting(num):

    m = biggie[num]

    mini = 1000

    for k in range(len(contours[m])):

        first = contours[m][k][0][0]

        if first < mini:

            mini = first

    mini1 = mini

    #print(mini1)



    mini = 1000

    for k in range(len(contours[m])):

        first = contours[m][k][0][1]

        if first < mini:

            mini = first

    mini2 = mini

   # print(mini2)



    maxi = 0

    for k in range(len(contours[m])):

        first = contours[m][k][0][1]

        if first > maxi:

            maxi = first

    maxi2 = maxi

    #print(maxi2)



    maxi = 0

    for k in range(len(contours[m])):

        first = contours[m][k][0][0]

        if first > maxi:

            maxi = first

    maxi1 = maxi

    #print(maxi1)

    ret = [mini2, maxi2, mini1, maxi1]

    s[num] = mini1

    return ret
fig, axs = plt.subplots(2, 5)

wow = plotting(-1)

axs[0, 0].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[0,0].axis('off')

wow = plotting(-2)

axs[0, 1].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[0,1].axis('off')

wow = plotting(-3)

axs[0,2].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[0,2].axis('off')

wow = plotting(-4)

axs[0,3].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[0,3].axis('off')

wow = plotting(-5)

axs[0,4].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[0,4].axis('off')

wow = plotting(-6)

axs[1,0].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[1,0].axis('off')

wow = plotting(-7)

axs[1,1].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[1,1].axis('off')

wow = plotting(-8)

axs[1,2].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[1,2].axis('off')

wow = plotting(-9)

axs[1,3].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[1,3].axis('off')

wow = plotting(-10)

axs[1,4].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

axs[1,4].axis('off')

wow = plotting(-11)

siggie = sorted(s, key=s.get)
def checkforerror(wow):

    white = cv2.countNonZero(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2])

    total = (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[0] * (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[1]

    div = white/total

    if div > 0.63: 

        if (wow[1] - wow[0]) - (wow[3] - wow[2]) > -5:

            return True

    elif div < 0.29:

        return True

    else: return False

def checkforerrorout(wow):

    white = cv2.countNonZero(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2])

    total = (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[0] * (binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2]).shape[1]

    div = white/total

    return div

def checkfordoubles(wow):

    if (wow[1] - wow[0]) - (wow[3] - wow[2]) < -4: 

        if checkforerrorout(wow) < 0.56: return True

    elif (wow[1] - wow[0]) - (wow[3] - wow[2]) < 1:

        if checkforerrorout(wow) < 0.40: return True

    else: return False
wow = plotting(siggie[0])

plt.imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

print('test 1', checkforerror(wow))

print((wow[1] - wow[0]) - (wow[3] - wow[2]) < -4)

print(checkforerrorout(wow))



print((wow[1] - wow[0]) - (wow[3] - wow[2]) < 1)

print('test 2', checkfordoubles(wow))
pos = 0

a = 0

im = 0

desired = 10

fig, axs = plt.subplots(2, 5)

first = False

while a < desired:

    print('a = ', a, 'desired = ', desired)

    

    n1 = pos // 5

    n2 = pos % 5

    print('using position', n1, n2)

    wow = plotting(siggie[im])

    print('trying to plot ', im, 'checking for errors...')

    err = checkforerror(wow)

    errs = checkfordoubles(wow)

    print('% = ', checkforerrorout(wow))

    if err:

        print('Error 1 at', a)

        #plt.imshow(binary[wow[0]-2:wow[1] + 2, wow[2]-2:wow[2] + half +7], cmap = 'gray')

        #inp = str(input('Skip this image? '))

        im += 1

        print('im bumped to', im)

        wow = plotting(siggie[im])

        print('trying to plot ', im, 'checking for errors...')

        err = checkforerror(wow)

        errs = checkfordoubles(wow)

        print('% = ', checkforerrorout(wow))

        if errs:

            

            

            print('Error 2 at bumped image')

            if first == False:

                half = abs(wow[0] - wow[1]) // 2

                axs[n1, n2].imshow(binary[wow[0]-2:wow[1] + 2, wow[2]-2:wow[2] + half +7], cmap = 'gray')

                xx = checkforerrorout(wow)

                axs[n1,n2].set_title(solution[pos]+' '+ str(int(xx * 100)))

                axs[n1,n2].axis('off')

                im -= 1

                first = True

            else:

                axs[n1, n2].imshow(binary[wow[0]-2:wow[1] + 2, wow[2] + half + 7:wow[3]+2], cmap = 'gray')

                xx = checkforerrorout(wow)

                axs[n1,n2].set_title(solution[pos]+' '+ str(int(xx * 100)))

                axs[n1,n2].axis('off')

                first = False



        

        

        else:

            print('No errors in bumped image')

            axs[n1, n2].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

            #xx = checkforerrorout(wow)

            axs[n1,n2].set_title(solution[pos]+' '+ str(int(xx * 100)))

            axs[n1,n2].axis('off')

    elif errs:

        print('Error 2 at', a)

        if first == False:

            half = abs(wow[0] - wow[1]) // 2

            axs[n1, n2].imshow(binary[wow[0]-2:wow[1] + 2, wow[2]-2:wow[2] + half +8], cmap = 'gray')

            xx = checkforerrorout(wow)

            axs[n1,n2].set_title(solution[pos]+' '+ str(int(xx * 100)))

            axs[n1,n2].axis('off')

            im -= 1

            first = True

        else:

            axs[n1, n2].imshow(binary[wow[0]-2:wow[1] + 2, wow[2] + half + 8:wow[3]+2], cmap = 'gray')

            xx = checkforerrorout(wow)

            axs[n1,n2].set_title(solution[pos]+' '+ str(int(xx * 100)))

            axs[n1,n2].axis('off')

            first = False

            

    else:

        print('No errors found!')

        axs[n1, n2].imshow(binary[wow[0]-2:wow[1]+2, wow[2]-2:wow[3]+2], cmap = 'gray')

        xx = checkforerrorout(wow)

        axs[n1,n2].set_title(solution[pos]+' '+ str(int(xx * 100)))

        axs[n1,n2].axis('off')

        print('DONE PLOTTING - im', im)



    a += 1

    pos += 1

    im += 1
plt.imshow(hi, cmap="gray")

plt.show()