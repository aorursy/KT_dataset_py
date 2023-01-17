import numpy as np
from skimage import color
from spams import trainDL, lasso

def beer_lambert_transform(tensor):
  ivec = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
  v = np.log(255) - np.log(ivec+1)

  # Remove white pixel from stain color map
  luminlayer = color.rgb2lab(tensor)[:,:,0]
  inew = ivec[ luminlayer.flatten() / np.max(luminlayer) < 0.9  , :]
  v_for_w = np.log(255) - np.log(inew+1)

  return v, v_for_w

def est_w_h(v, v_initial, n_stains, sparsity_thr):
  w = trainDL(np.asfortranarray(v_initial.T),mode=2,iter=100,lambda1=sparsity_thr,posAlpha=True,posD=True,modeD=0,whiten=False,K=n_stains,verbose=False)
  #w = w[w[:,0].argsort()]
  H = lasso(np.asfortranarray(v.T),np.asfortranarray(w),mode=2,lambda1=sparsity_thr,pos=True,verbose=False)
  return H, w

def stain_seperator(img, n_stains, sparsity_thr=0.8, return_wh=False):
  assert np.ndim(img) == 3, "stain_seperator only works with rgb images"                                                                    
  assert img.shape[2] == 3, "stain_seperator expects image with shape [.., .., 3]"                                                          

  if img.dtype.char not in np.typecodes['AllFloat']:                  
    img = img.astype(np.float)     

  width = img.shape[0]             
  height = img.shape[1]            

  (v, v_for_w) = beer_lambert_transform(img)                          
  h, w = est_w_h(v, v_for_w, n_stains, sparsity_thr)                  

  if return_wh:                    
      return w, h.toarray()        

  h1 = w*h                         
  h1 = np.exp(-h1)                 

  stain_maps = np.moveaxis(h1.reshape(3,width,height),0,-1)           

  return stain_maps                

