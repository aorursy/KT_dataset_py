def read_image(filepath):

    im = io.imread(filepath)

    # scale

    (w, h,_) = im.shape

    if (w>h): # height is greater than width

        resizeto = (IMAGE_SIZE, int (round (IMAGE_SIZE * (float (h)  / w))));

    else:

        resizeto = (int (round (IMAGE_SIZE * (float (w)  / h))), IMAGE_SIZE);

    im = misc.imresize(im, resizeto, interp='bicubic').astype(np.float32) # it's float32 as of now

    # swap RBG->BGR

    im = im[:, :, ::-1]

    

    # remove mean (sueezenet mean values)

    im[:, :, 0] -= 104.006

    im[:, :, 1] -= 116.669

    im[:, :, 2] -= 122.679



    

    # padd with 0 (since we're already mean centered the 0 is correct value for padding)

    paddim=np.array((IMAGE_SIZE,IMAGE_SIZE))-np.array(resizeto)

    npad=((0,paddim[0]),(0,paddim[1]),(0,0))

    im = np.pad(im,npad, mode='constant')

    return im



def unprocess_image(im):   

    # restore mean (sueezenet mean values)

    im[:, :, 0] += 104.006

    im[:, :, 1] += 116.669

    im[:, :, 2] += 122.679

    

    # swap RBGR->RBG

    im = im[:, :, ::-1]

    

    return im