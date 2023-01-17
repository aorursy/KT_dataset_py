from keras.preprocessing import image



# u can play with the parameters of the augmentation, there are mant more

image_datagen = image.ImageDataGenerator(height_shift_range=0.2 , horizontal_flip=True , rotation_range=15 ,width_shift_range=0.2 ,shear_range=0.2)

mask_datagen = image.ImageDataGenerator(height_shift_range=0.2 , horizontal_flip=True , rotation_range=15 ,width_shift_range=0.2 ,shear_range=0.2)

#the seed is what determines the random augmentation so u can change it but keep it is the same in the

#mask and image generators

seed = 1

image_datagen.fit(X_train, augment=True, seed=seed)

mask_datagen.fit(Y_train, augment=True, seed=seed)



x=image_datagen.flow (X_train,batch_size=15,shuffle=True,seed=seed)

y=mask_datagen.flow (Y_train,batch_size=15,shuffle=True,seed=seed)





#u can see here if the generators are sychronized and check the images and masks they produce



from matplotlib import pyplot as plt

%matplotlib inline



imshow(x.next()[0].astype (np.uint8))

plt.show()

imshow(np.squeeze (y.next()[0].astype (np.uint8)))

plt.show()
#creating one generator that generates masks and images

train_generator = zip(x, y)



model.fit_generator(

    train_generator,

    steps_per_epoch=60,

    epochs=200 )