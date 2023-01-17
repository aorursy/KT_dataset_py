import cv2

import numpy as np

from matplotlib import pyplot as plt
image_path = '../input/experiment-images/image.jpg'

image = cv2.imread(image_path)

image = image[:,:,::-1]

plt.imshow(image)
def nst(trained_model_path,image):



    model = trained_model_path

    net = cv2.dnn.readNetFromTorch(model)



    image = cv2.imread(image)

    (h,w) = image.shape[:2]



    blob = cv2.dnn.blobFromImage(image, 1.0, (w,h),

            (103.939, 116.779, 123.680), swapRB = False, crop = False)



    net.setInput(blob)

    output = net.forward()



    output = output.reshape((3,output.shape[2],output.shape[3]))

    output[0] += 103.939

    output[1] += 116.779

    output[2] += 123.680

    

    output = output.transpose(1,2,0)

    output = np.clip(output, 0, 255)

    output= output.astype('uint8')



    return output 
path = '../input/trained-model-for-fast-nst/candy.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/composition_vii.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/feathers.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/la_muse.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/mosaic.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/starry_night.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/the_scream.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/the_wave.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)
path = '../input/trained-model-for-fast-nst/udnie.t7'

output = nst(path, image_path)

output = output[:,:,::-1]

plt.imshow(output)