import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
img = mpimg.imread('../input/home.png')

data = img.reshape(1,img.shape[0], img.shape[1], img.shape[2])





print(type(img))

print("Image dimension ",img.shape)

print("Input data dimension ", data.shape)
plt.imshow(data[0,:,:,:])

plt.grid(False)

plt.axis("off")
def zero_pad(data, pad):

    data_padded = np.pad(array = data, pad_width = ((0,0),(pad,pad), (pad,pad), (0,0)), mode = 'constant', constant_values = 0)

    return data_padded

    
print("dimension before padding: ", data.shape)

img_pad = zero_pad(data, 10)

print("dimension after padding: ", img_pad.shape)

print(img_pad[0,8:12,8:12,1])

plt.imshow(img_pad[0,:,:,:], cmap = "gray")

plt.grid(False)



output1 = np.mean(img_pad)
def conv_single_step(data_slice, W, b):

    conv = np.multiply(data_slice, W)

    Z = np.sum(conv) + b

    

    return Z
def conv_forward(A_prev, W, b, hparams):

  stride = hparams["stride"]

  pad = hparams["pad"]

 

  m, h_prev, w_prev, c_prev = A_prev.shape

   

  f, f, c_prev, n_c = W.shape

  

  n_h = int((h_prev - f + 2*pad)/stride) + 1

  n_w = int((w_prev - f + 2*pad)/stride) + 1

  

  Z = np.zeros((m, n_h, n_w, n_c))

  A_prev_pad = zero_pad(A_prev, pad)

  for i in range(m):

    for h in range(n_h):

      for w in range(n_w):

        for c in range(n_c):

           w_start = w * stride

           w_end = w_start + f 

           h_start = h * stride

           h_end = h_start + f

        

           Z[i,h,w,c] = conv_single_step(A_prev_pad[i, h_start:h_end, w_start:w_end, :], W[:,:,:,c], b[:,:,:,c])

  return Z
np.random.seed(1)

input_ = np.random.randn(10, 4, 4, 3)

W = np.random.randn(2, 2, 3, 8)

b = np.random.randn(1, 1, 1, 8)

hparameters = {"pad" : 1,

               "stride": 1}



output_ = conv_forward(input_, W, b, hparameters)

print(np.mean(output_))
edge_detect = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).reshape((3,3,1,1))


hparams = {"pad" : 0,

               "stride": 1}

b = np.zeros((1, 1, 1, 1))

Z = conv_forward(data, edge_detect, b, hparams)





plt.clf()

plt.imshow(Z[0,:,:,0], cmap='gray',vmin=0, vmax=1)

plt.grid(False)

print("dimension of image before convolution: ", data.shape)

print("dimension of image after convolution: ", Z.shape)



output2 = np.mean(Z[0,100:200,200:300,0])





##below are the filters for vetical as well as horizontal edge detection, try these filters once you have completed this handson.

vertical_filter = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]]).reshape(3,3,1,1)



Z = conv_forward(data, vertical_filter, b, hparams)

plt.clf()

plt.imshow(Z[0,:,:,0], cmap='gray',vmin=0, vmax=1)

plt.grid(False)

print("dimension of image before convolution: ", data.shape)

print("dimension of image after convolution: ", Z.shape)





horizontal_filter = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]).reshape((3,3,1,1))

Z = conv_forward(data, horizontal_filter, b, hparams)

plt.clf()

plt.imshow(Z[0,:,:,0], cmap='gray',vmin=0, vmax=1)

plt.grid(False)

print("dimension of image before convolution: ", data.shape)

print("dimension of image after convolution: ", Z.shape)

def max_pool(input, hparam):

    m, h_prev, w_prev, c_prev = input.shape

    f = hparam["f"]  ## f is the filter size to use for pooling

    stride = hparam["stride"]

    h_out = int(((h_prev - f)/stride) + 1)

    w_out = int(((w_prev -f)/stride) + 1)

    output = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):

        for c in range(c_prev):

            for h in range(h_out):

                for w in range(w_out):

                    w_start = w * stride

                    w_end = w_start + f

                    h_start = h * stride

                    h_end = h_start + f

                    output[i, h, w, c] = np.max(input[i,h_start:h_end, w_start:w_end, c])

    print(output.shape)

    assert output.shape == (m, h_out, w_out, c_prev)

    return output
pool_params = {"stride" : 2, "f" : 2}

output_ = max_pool(input_, pool_params)

print(np.mean(output_))
###start code

hparams = {'stride':1, 'f':2}

Z_pool =  max_pool(input_, hparams)

###End code



print("dimension before pooling :", Z.shape)

print("dimension after pooling :", Z_pool.shape)



plt.imshow(Z_pool[0,:,:,0], cmap = "gray")



with open("output.txt", "w+") as file:

    file.write("output1 = %f" %output1)

    file.write("\noutput2 = %f" %output2)