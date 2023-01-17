import matplotlib.pyplot as plt



import matplotlib.image as mimg



import numpy as np
samp = 20



train_data = np.zeros((20*5, 144*120*3))



train_label = np.zeros((20*5))



count = -1



plt.figure(1)



plt.ion()
for i in range(1, 6):

    

    for j in range(1, samp + 1):

    

        plt.cla()

    

        count = count + 1

        

        path = '../input/facesamples-dataset/facesample/facesamples/a%d/%d.png'%(i, j)

    

        train_image = mimg.imread(path)

    

        train_feat = train_image.reshape(1, -1)

    

        train_data[count, :] = train_feat

    

        train_label[count] = i



    

        plt.imshow(train_image, cmap = 'gray')

    

        plt.title('Actress No.::  ' + str(i) + '  ::  Sample No.::  ' + str(j))

    

        plt.pause(0.1)
test_data = np.zeros((1, 144*120*3))



test_label = np.zeros((1))



count = -1



a = 11



print("Enter The Number To Select Test Image from 1 t0 15 ::  ", a)
for i in range(a, a + 1):

    

    path = '../input/facesamples-dataset/facesample/testdata/%d.png'%i

  

    test_image = mimg.imread(path)

  

    test_feat = test_image.reshape(1, -1)

  

    test_data[count, :] = test_feat

  

    test_label[count] = i



    plt.imshow(test_image, cmap = 'Blues')

  

    plt.title('The Test Image')

for i in range(0, 1):

    

    test_sample = test_data[i, :]

    

    actual_label = test_label[i]



    d = np.zeros((samp*5))



    for j in range(0,samp*5):

    

        d[j] = np.sum((test_sample-train_data[j, :])**2)



    val = d.min()

    

    ind = d.argmin()

    

    predicted_value = int(train_label[ind])
if predicted_value == 1:

    

    print('This picture belongs to Aishwarya Rai Bachchan')



if predicted_value == 2:



    print('This picture belongs to Kajal Aggarwal')



if predicted_value == 3:



    print('This picture belongs to Shriya Saran')



if predicted_value == 4:



    print('This picture belongs to Rashmika Mandanna')



if predicted_value == 5:



    print('This picture belongs to Tamanna Bhatia')

    

plt.imshow(test_image, cmap = 'gray')



plt.title('The Test Image')



plt.pause(0.1)

    

location = ind - (samp*(predicted_value - 1))

    

predicted_data = np.zeros((1, 144*120*3))



predicted_label = np.zeros((1))



count = -1



for i in range(predicted_value, predicted_value + 1):

    

    for j in range(location, location + 1):

        

        count = count + 1

    

        path = '../input/facesamples-dataset/facesample/facesamples/a%d/%d.png'%(i, j)

    

        predicted_image = mimg.imread(path)

  

        predicted_feat = predicted_image.reshape(1, -1)

  

        predicted_data[count, :] = predicted_feat

      

        predicted_label[count] = i



        plt.imshow(predicted_image, cmap = 'Blues')

        

        plt.title('The Predicted Image \n Actress No.::  ' + str(predicted_value) + '  ::  Sample No.::  ' + str(ind))

    