# Following are the modules we will need.
import matplotlib.pyplot as plt
import imageio
import copy
import numpy as np

%matplotlib notebook
myimage = imageio.imread('../input/arctic-ice-images-data/myIceImage.jpg') 
myImage2 = imageio.imread('../input/arctic-ice-images-data/IceImage2.jpg')

plt.figure()
plt.subplot(121)
plt.imshow(myimage)
#plt.axis('off')     # Remove axis ticks and numbers
plt.axis('image')    # Set aspect ratio to obtain square pixels.

plt.subplot(122)
plt.imshow(myImage2)
#plt.axis('off')     # Remove axis ticks and numbers
plt.axis('image')    # Set aspect ratio to obtain square pixels.
# Show a few colors
plt.figure()
plt.subplot(221); 
plt.imshow(myImage2[40:70,-70:-40,:]); plt.axis('off'); plt.axis('image'); plt.title(str(myImage2[40,-70,:]))

plt.subplot(222); 
plt.imshow(myImage2[380:410,330:360,:]); plt.axis('off'); plt.axis('image'); plt.title(str(myImage2[395,345,:]))

plt.subplot(223); 
plt.imshow(myImage2[530:550,450:470]); plt.axis('off'); plt.axis('image'); plt.title(str(myImage2[540,460,:]))

plt.subplot(224); 
plt.imshow(myImage2[570:590,520:540,:]); plt.axis('off'); plt.axis('image'); plt.title(str(myImage2[580,530,:]))


# Show a few more colors
plt.figure()
plt.subplot(221); 
plt.imshow(myImage2[590:620,150:180,:]); plt.axis('off'); plt.axis('image'); plt.title(str(myImage2[605,165,:]))

plt.subplot(222); 
plt.imshow(myImage2[166:176,618:621,:]); plt.axis('off'); plt.axis('image'); plt.title(str(myImage2[170,619,:]))

plt.subplot(223); 
plt.imshow(myImage2[170:182,590:602,:]); plt.axis('off'); plt.axis('image'); plt.title('Various')

plt.subplot(224); 
plt.imshow(myImage2[660:680,54:74,:]); plt.axis('off'); plt.axis('image'); plt.title('Various')

# Finding indices is easier if we reshape the 3-D matrix into a 2-D matrix, where 
# the first dimension is each row in order and the second is the color
r = np.ravel(myImage2[:,:,0])
g = np.ravel(myImage2[:,:,1])
b = np.ravel(myImage2[:,:,2])

# A function for getting indices to the matrix that are all within a range
def getIndicesForColorRanges(r,g,b,redrange,greenrange,bluerange):
    
    ired = np.where(np.logical_and(r>=redrange[0], r<=redrange[1]))[0]
    igrn = np.where(np.logical_and(g>=greenrange[0], g<=greenrange[1]))[0]
    iblu = np.where(np.logical_and(b>=bluerange[0], b<=bluerange[1]))[0]
                    
    ind = np.intersect1d(np.intersect1d(ired,igrn),iblu)

    return ind


# black: 15, 1, 1,
# green: 35, 132, 101
# red: 241, 43, 70
# pink: 228, 155, 128
# white: 190, 188, 175
# latlon: 148, 133, 188

# red => ice  
# orange => snow  
# white to dark peach => clouds  
# bright blue => lines of latitude and longitude
# green => bare ground
# black => ocean  


# Choose unique colors
iblack  = getIndicesForColorRanges(r, g, b, [  0, 125], [  0,  80], [   0,  80])   # black
iblue   = getIndicesForColorRanges(r, g, b, [  0, 125], [  0,  80], [  81, 250])   # blue
igreen1 = getIndicesForColorRanges(r, g, b, [  0, 125], [ 81, 125], [   0, 250])   # green
igreen2 = getIndicesForColorRanges(r, g, b, [  0, 125], [126, 250], [   0, 250])   # green-blue

iice    = getIndicesForColorRanges(r, g, b, [126, 250], [  0, 125], [  0,  80]) # red
ipink1  = getIndicesForColorRanges(r, g, b, [126, 250], [  0, 125], [ 81, 125]) # pink
ipurple = getIndicesForColorRanges(r, g, b, [126, 250], [  0, 125], [126, 250]) # purple
igray   = getIndicesForColorRanges(r, g, b, [126, 250], [126, 250], [  0, 125]) # gray
iwhite1 = getIndicesForColorRanges(r, g, b, [126, 200], [126, 250], [126, 200]) # white/gray
ilatlon = getIndicesForColorRanges(r, g, b, [126, 200], [126, 250], [201, 250]) # ???

ipink2   = getIndicesForColorRanges(r, g, b, [201, 250], [126, 150], [126, 182]) # pink
iwhite2  = getIndicesForColorRanges(r, g, b, [201, 250], [126, 150], [183, 250]) # white
iwhite3  = getIndicesForColorRanges(r, g, b, [201, 250], [151, 195], [126, 182]) # white
ipink3   = getIndicesForColorRanges(r, g, b, [201, 250], [126, 195], [183, 200]) # pink
iwhite4  = getIndicesForColorRanges(r, g, b, [201, 250], [126, 195], [201, 250]) # ???
iwhite5  = getIndicesForColorRanges(r, g, b, [201, 250], [196, 250], [183, 250]) # white

# Choose scene types
iground = np.union1d(np.union1d(iblue,igreen1),igreen2)
icloud  = np.union1d(np.union1d(np.union1d(np.union1d(np.union1d(iwhite1,iwhite2),iwhite3),iwhite4),iwhite5),igray)
icloud2 = np.union1d(np.union1d(ipink1,np.union1d(ipink2,ipink3)),ipurple)
iocean  = iblack



# Reshape as a matrix with one vector for each color
myImage3 = copy.deepcopy(myImage2).reshape((646400,3))

# Make an all-red image
myImage3[:,0] = 150; myImage3[:,1] = 150; myImage3[:,2] = 150  # Background = grey

# Now color the image according to the indices
myImage3[iocean,:] = [0, 0, 250]        # Color Ocean blue
myImage3[iground,:] = [0, 250, 0]       # Land = blues and greens
myImage3[iice,:]    = [250, 250, 0]     # Color ice yellow
myImage3[icloud,:]  = [250, 250, 250]   # Color cloud white
myImage3[icloud2,:] = [200, 200, 200]   # Color pink cloud light gray
myImage3[ilatlon,:] = [0, 0, 0]       # Color lines of latitude and longitude black

myImage3 = myImage3.reshape(808,800,3)

plt.figure()
plt.subplot(121)
plt.imshow(myImage2)
plt.axis('off'); plt.axis('image')   

plt.subplot(122)
plt.imshow(myImage3)
plt.axis('off'); plt.axis('image')   

# Now color the image according to the indices
Npixels = myImage2.shape[0]*myImage2.shape[1] +0.  
print('Fraction ocean: ' + str(len(iocean)/Npixels))
print('Fraction bare ground: ' + str(len(iground)/Npixels))
print('Fraction ice: ' + str(len(iice)/Npixels))
print('Fraction cloud: ' + str((len(icloud)+len(icloud2))/Npixels))
print('Fraction lines of latitude/longitude: ' + str(len(ilatlon)/Npixels))
print('Fraction unidentified: ' + str(1-
       (len(iocean)+len(iground)+len(iice)+len(icloud)+len(icloud2)+len(ilatlon))/Npixels))



