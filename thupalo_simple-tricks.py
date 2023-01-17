# resize image (101,101) -> (128,128)
def image_in(img):    
    out = np.zeros((128,128), dtype=np.uint8)    
    ih = img[:, ::-1] 
    iv = img[::-1, :]
    ihv= ih[::-1, :] 
    out[13:114, 13:114] = img    
    out[0:13,0:13] = ihv[-13:,-13:]
    out[0:13,13:114] = iv[-13:,:]
    out[0:13, -14:] = ihv[-13:, 0:14]
    out[13:114, 0:13] = ih[:,-13:]
    out[-14:, 0:13] = ihv[0:14:,-13:]
    out[-14:,13:114] = iv[0:14,:]
    out[-14:,-14:] = ihv[0:14,0:14]
    out[13:114,-14:]  =ih[:,0:14]
    return(out)

# go back to original size (128,128) -> (101,101) 
def image_out(img):
    return img[13:114,13:114]   # for 4 dims use [:,13:114,13:114,:]
# instead using heavy augmentations, 
# flip left-right is simple and maybe is the only method allowed
def flip():
    X_aug = X_train[:,:,::-1,:]
    y_aug = y_train[:,:,::-1,:]
# this is tricky
# having keras model trained, even with augmented left-right data, 
# try predict with normal and mirrored X_test and take the average
def dummy_prediction():
    # ...
    y_preds = keras_model.predict(X_test, verbose=0)
    if mirror==True:
        m_preds = keras_model.predict(X_test[:,:,::-1,:], verbose=0)
        y_preds = 0.5 * (y_preds + m_preds[:,:,::-1,:])
    # ...

# check how much you gain, if yes please upvote
