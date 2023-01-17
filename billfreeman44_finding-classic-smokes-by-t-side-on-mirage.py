import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
csgo = pd.read_csv('../input/mm_grenades_demos.csv')
index=[(csgo['nade']=='Smoke') & 
       (csgo['map']=='de_mirage') & 
       (csgo['att_side'] == 'Terrorist')][0]
csgo_sm=csgo[index]
csgo_sm_small=csgo_sm[[ 'nade_land_x',
                       'nade_land_y',
                       'att_rank', 
                       'att_pos_x', 
                       'att_pos_y']]
csgo_sm_small.dropna()
def pointx_to_resolutionx(xinput,startX=-3217,endX=1912,resX=1024):
    sizeX=endX-startX
    if startX < 0:
        xinput += startX *(-1.0)
    else:
        xinput += startX
    xoutput = float((xinput / abs(sizeX)) * resX);
    return xoutput

def pointy_to_resolutiony(yinput,startY=-3401,endY=1682,resY=1024):
    sizeY=endY-startY
    if startY < 0:
        yinput += startY *(-1.0)
    else:
        yinput += startY
    youtput = float((yinput / abs(sizeY)) * resY);
    return resY-youtput
csgo_sm_small['thrower_xpos']=csgo_sm_small['att_pos_x'].apply(pointx_to_resolutionx)
csgo_sm_small['thrower_ypos']=csgo_sm_small['att_pos_y'].apply(pointy_to_resolutiony)
csgo_sm_small['nade_ypos']=csgo_sm_small['nade_land_y'].apply(pointy_to_resolutiony)
csgo_sm_small['nade_xpos']=csgo_sm_small['nade_land_x'].apply(pointx_to_resolutionx)
csgo_sm_small.head() #looks like it worked fine...
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(11,11))
t = plt.imshow(im)
t = plt.scatter(csgo_sm_small['nade_xpos'], csgo_sm_small['nade_ypos'],alpha=0.05,c='blue')
t = plt.scatter(csgo_sm_small['thrower_xpos'], csgo_sm_small['thrower_ypos'],alpha=0.05,c='red')
#Drop old raw columns so we don't get confused
csgo_sm_small.drop(['nade_land_x','nade_land_y','att_pos_x','att_pos_y'],inplace=True,axis=1)
csgo_sm_small.columns
#rename my frame css cause its too long
css=csgo_sm_small
def calc_N_Nearby(x,y,xarr,yarr,dt=15):
    index=[(xarr < (x + dt)) & (xarr > (x - dt)) & (yarr < (y + dt)) & (yarr > (y - dt))][0]
    return len(xarr[index])

zarr15=[]
#zarr10=[] #I already know I want density 15 but others could be useful...
#zarr5=[]
#takes a hot minute
for i in range(len(css['thrower_xpos'])):
    zarr15.append(calc_N_Nearby(css['thrower_xpos'].iloc[i],css['thrower_ypos'].iloc[i],
              css['thrower_xpos'],css['thrower_ypos'],dt=15))
    #zarr10.append(calc_N_Nearby(css['thrower_xpos'].iloc[i],css['thrower_ypos'].iloc[i],
    #          css['thrower_xpos'],css['thrower_ypos'],dt=10))
    #zarr5.append(calc_N_Nearby(css['thrower_xpos'].iloc[i],css['thrower_ypos'].iloc[i],
    #          css['thrower_xpos'],css['thrower_ypos'],dt=5))
    #print(z)
    


css['density15']=zarr15
x=plt.hist(zarr15,bins=75)

index=[(css['density15']>100)][0]#I use this notation cause I'm used to a language called IDL and it works like this
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(12,12))
implot = plt.imshow(im)
implot = plt.scatter(css['thrower_xpos'][index], css['thrower_ypos'][index],alpha=0.15
                     ,c='red',s=5) 
implot = plt.scatter(css['nade_xpos'][index], css['nade_ypos'][index],alpha=0.15
                     ,c='blue',s=5) 
index=[(css['density15']>100)][0]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=25) #n clusters required??
kmeans.fit(css[["thrower_xpos","thrower_ypos"]][index])
c_centers=pd.DataFrame(kmeans.cluster_centers_)
im = plt.imread('../input/de_mirage.png')
plt.figure(figsize=(12,12))
implot = plt.imshow(im)
implot = plt.scatter(css['thrower_xpos'][index], css['thrower_ypos'][index],alpha=0.25
                     ,c='red',s=5) 
implot = plt.scatter(css['nade_xpos'][index], css['nade_ypos'][index],alpha=0.25
                     ,c='blue',s=5) 

implot = plt.scatter(c_centers[0],c_centers[1],alpha=1.0
                     ,c='green',s=35) 
#perhaps the worst way to do this?
css['kmeanslabel']=np.zeros(len(index))
css['kmeanslabel'][index]=kmeans.labels_
#for i in range(0,27): #causes max figure opened error/warning...
for i in range(1,3):
    index=[css['kmeanslabel'] == i][0]
    im = plt.imread('../input/de_mirage.png')
    plt.figure(figsize=(12,12))
    implot = plt.imshow(im)
    implot = plt.scatter(css['thrower_xpos'][index], css['thrower_ypos'][index],alpha=0.25
                         ,c='red',s=5) 
    implot = plt.scatter(css['nade_xpos'][index], css['nade_ypos'][index],alpha=0.25
                         ,c='blue',s=5) 
    #implot = plt.savefig("picked_out_nades_"+str(i)+".png")


