from keras.applications import InceptionV3, ResNet50, MobileNetV2, Xception
model = InceptionV3()

model.save("InceptionV3.h5")
model = ResNet50()

model.save("ResNet50.h5")
model = MobileNetV2()

model.save("MobileNetV2.h5")
model = Xception()

model.save("Xception.h5")