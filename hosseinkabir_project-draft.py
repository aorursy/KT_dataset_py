import numpy as np
import pandas as pd
fd = '../input/cee-498-project8-pore-in-concrete/train.xlsx'

# tabular data for first and second batches are placed in first and second sheets of our excel sheet, respectively

batch1 = pd.DataFrame(pd.read_excel(fd, sheet_name = 0))
batch2 = pd.DataFrame(pd.read_excel(fd, sheet_name = 'Batch2'))
from IPython.core.display import display, HTML

# set style of table caption

styles = [dict(selector="caption", 
    props=[("text-align", "center"),
    ("font-size", "120%"),
    ("color", 'blue')])]

# this block is partially adapted from https://stackoverflow.com/a/58866040

# prints the tables (i.e. batch1 and batch2) side by side.

def display_side_by_side(dfs:list, captions:list):
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline; font-size:90%;color:black;textalign:center'"
             ).set_caption(caption).set_table_styles(styles)._repr_html_()

# increasing spacing between two tables

        output += "\xa0\xa0\xa0\xa0\xa0\xa0"
    display(HTML(output))
    
# assign suitable caption to each batch

batch1.rename(columns=({'ID':'batch1_image_id'}), inplace=True); batch2.rename(columns=({'ID':'batch2_image_id'}), inplace=True);
display_side_by_side([batch1.head(5),batch2.head(5)],['Batch1', 'Batch2'])

print('batches 1 and 2 have %s and %s images/rows, respectively.'%(len(batch1),len(batch2)))
from matplotlib import ticker as tick
from matplotlib.figure import Figure
import PIL
import matplotlib.pyplot as plt
import glob

# representing first five images of batch 1 in subplot format

fig, axes=plt.subplots(nrows=1, ncols=5, figsize=(20,20))

# the format of images were stored in png format 

# we select the first 5 images of batch1 just to show the reader how they look like in viridis threshold system

for i, f in enumerate(glob.glob('../input/cee-498-project8-pore-in-concrete/batch1/batch1/*.png')[:5]):
    ax=axes[i%5]
    ax.set_title('batch1_image_%d'%(i+1),fontsize=15, color='r', pad=15)
    
# values shown on x and y axes represents number of pixels in orthogonal directions 

    ax.set_xlabel('width',fontsize=15)
    ax.set_ylabel('height',fontsize=15)
    im=PIL.Image.open(f)
    ax.imshow(im);
    fig.subplots_adjust(wspace=20)
    fig.tight_layout(pad=-4)
print('dimension of each image is %s (width) by %s (height) pixels'%((im.size)))
fig, axes=plt.subplots(ncols=2, figsize=(20,6))

#  here we illustrate two completely different color scales for image analysis

for k in range(2):
        img = np.zeros((100,256),dtype=np.uint8)
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                img[j,i]=i
                
#  Viridis color scale

        axes[0].imshow(img, cmap = 'viridis')
    
#  Greyscale color scale
    
        axes[1].imshow(img, cmap = 'gray')
        axes[k].tick_params(axis='y',left = False,labelleft = False)
        
#  numbers appeared on x-axis tick-mark increases every 50 threshold

        axes[k].set_xticks([0,50,100,150,200,255])
        axes[k].tick_params(labelsize=16)
        axes[0].set_title('Viridis threshold', fontsize=19)
        axes[1].set_title('Greyscale threshold', fontsize=19)
# converting Viridis to Greyscale thresholds

fig, axes=plt.subplots(nrows=1, ncols=5, figsize=(20,20))
for i, f in enumerate(glob.glob('../input/cee-498-project8-pore-in-concrete/batch1/batch1/*.png')[:5]):
    ax=axes[i%5]
    ax.set_title('batch1_greyscale_image_%d'%(i+1),fontsize=15, color='r', pad=15)
    ax.set_xlabel('width',fontsize=15)
    ax.set_ylabel('height',fontsize=15)
    im=PIL.Image.open(f)
    ax.imshow(im.convert('LA'));
    fig.subplots_adjust(wspace=10)
    fig.tight_layout(pad=-10)
from PIL import Image
import requests
from io import BytesIO
plt.figure(figsize=(10,7))

# importing sample image from open web source

sample_image = requests.get('https://www.zkg.de/imgs/1/5/3/9/6/9/8/Schade_Bild_8a-9d4e1097c0899084.jpeg')
sample_back_scattered_image = Image.open(BytesIO(sample_image.content))
plt.imshow(sample_back_scattered_image);
ave_porosity_batch1=[]
porosity_batch1_sub_threshold=[]

# greyscale threshold of images in batch1 varies from 0 to 40

threshold_limit=41
for i in range(threshold_limit):
    for f in glob.glob('../input/cee-498-project8-pore-in-concrete/batch1/batch1/*.png'):
        im_bch1=PIL.Image.open(f).convert('LA')
        total=np.array(im_bch1)
        poros_bch1_count=0
        for j in total:
            for k in j:
                
# we quantify pores having luminance threshold less than i

                if float(k[0]) < (i+1):
                    poros_bch1_count+=1            
        porosity_batch1_sub_threshold.append(poros_bch1_count/(im.size[0]*im.size[1]))
        
# for different thresholds we calculate ave porosity of batch1 to compare them with exact ave porosity of this batch

    ave_porosity_batch1.append(np.mean(porosity_batch1_sub_threshold))
ave_porosity_batch2=[]
porosity_batch2_sub_threshold=[]

# greyscale threshold of images in batch2 varies from 0 to 40

threshold_limit=41
for i in range(threshold_limit):
    for f in glob.glob('../input/cee-498-project8-pore-in-concrete/batch2/batch2/*.png'):
        im_bch2=PIL.Image.open(f).convert('LA')
        total=np.array(im_bch2)
        poros_bch2_count=0
        for j in total:
            for k in j:
                
# we quantify pores having luminance threshold less than i

                if float(k[0]) < (i+1):
                    poros_bch2_count+=1            
        porosity_batch2_sub_threshold.append(poros_bch2_count/(im.size[0]*im.size[1]))
        
# for different thresholds we calculate ave porosity of batch2 to compare them with exact ave porosity of this batch

    ave_porosity_batch2.append(np.mean(porosity_batch2_sub_threshold))
gray_scale_threshold=[]
for i in range(threshold_limit):
    gray_scale_threshold.append(i)
    
# ave porosities are multiplied by 100 to be expressed in percentage(%)

batch_data={'Batch1 ave porosity(%)':100*np.array(ave_porosity_batch1),'Batch2 ave porosity(%)':100*np.array(ave_porosity_batch2),
             'threshold':np.array(gray_scale_threshold)}
df_batch=pd.DataFrame(batch_data)
df_batch[30:41]
# illustrating subplots of ave porosity for each batch as a funciton of threshold limit

fig, axes=plt.subplots(nrows=1, ncols=2, figsize=(20,6))

# batch1

axes[0].scatter(df_batch['Batch1 ave porosity(%)'], df_batch['threshold'], color='red');
axes[0].set_xlabel("batch1 ave porosity(%)", size=16);axes[0].set_ylabel("greyscale threshold", size=16);axes[0].set_title(
        "Batch1 ave porosity vs. greyscale threshold", fontsize=16,color='blue');
plt.setp(axes[0].get_xticklabels(), fontsize=12);plt.setp(axes[0].get_yticklabels(), fontsize=12);

# batch2  

axes[1].scatter(df_batch['Batch2 ave porosity(%)'], df_batch['threshold'], color='red');
axes[1].set_xlabel("batch2 ave porosity(%)", size=16);axes[1].set_ylabel("greyscale threshold", size=16);axes[1].set_title(
        "Batch2 ave porosity vs. greyscale threshold", fontsize=16, color='blue');
plt.setp(axes[1].get_xticklabels(), fontsize=12);plt.setp(axes[1].get_yticklabels(), fontsize=12);
import cv2
fig, axes=plt.subplots(nrows=1, ncols=5, figsize=(20,20))
for i in range(5):
        img = cv2.imread('../input/cee-498-project8-pore-in-concrete/batch1/batch1/image_1_7.png')

# changing luminance of pixels having a threshold limit < 10*i to black (and the rest to white)

        img[img >10*i] = 255
        ax=axes[i%5]
        ax.set_title('batch1_image_1_7_threshold= %d'%(10*(i)),fontsize=15, color='r', pad=15)
        ax.set_xlabel('width',fontsize=15)
        ax.set_ylabel('height',fontsize=15)
        ax.imshow(img)
        fig.subplots_adjust(wspace=10)
        fig.tight_layout(pad=-15)
fig, axes=plt.subplots(nrows=1, ncols=5, figsize=(20,20))
for i in range(5):
        img = cv2.imread('../input/cee-498-project8-pore-in-concrete/batch2/batch2/image_2_2.png')
        
# changing luminance of pixels having a threshold limit < 10*i to black (and the rest to white)
        
        img[img >10*i] = 255
        ax=axes[i%5]
        ax.set_title('batch2_image_1_7_threshold= %d'%(10*(i)),fontsize=15, color='r', pad=15)
        ax.set_xlabel('width',fontsize=15)
        ax.set_ylabel('height',fontsize=15)
        ax.imshow(img)
        fig.subplots_adjust(wspace=10)
        fig.tight_layout(pad=-15)
import os
bacth1_image_id=[]
df_first_id_to_integer=[]
df_second_id_split=[]

# importing the name of images in batch1

for f in glob.glob('../input/cee-498-project8-pore-in-concrete/batch1/batch1/*.png'):
        bacth1_image_id.append(os.path.split(f)[-1])
measured_porosity=pd.DataFrame({'batch1_image_id':np.array(bacth1_image_id)})

# sorting images to compare calculated porosities in batch1 with exact porosity

df_first_id=measured_porosity['batch1_image_id'].str.split('_', expand=True)[1]
for i in df_first_id:
    df_first_id_to_integer.append(int(i))
df_second_id=measured_porosity['batch1_image_id'].str.split('_', expand=True)[2]
for i in df_second_id:
    df_second_id_split.append(int(''.join(list(filter(str.isdigit, i)))))

# representing porosity of each image at different thresholds

measured_porosity_splitted_batch1=pd.DataFrame({'batch1_img_1st_id':df_first_id_to_integer, 'batch1_img_2nd_id':df_second_id_split,'posority_thershold=0'
                                :100*np.array(porosity_batch1_sub_threshold)[0:100],'posority_thershold=10'
                                :100*np.array(porosity_batch1_sub_threshold)[1000:1100],'posority_thershold=13'
                                :100*np.array(porosity_batch1_sub_threshold)[1300:1400],'posority_thershold=14'
                                :100*np.array(porosity_batch1_sub_threshold)[1400:1500],'posority_thershold=20'
                                :100*np.array(porosity_batch1_sub_threshold)[2000:2100],'posority_thershold=30'
                                :100*np.array(porosity_batch1_sub_threshold)[3000:3100],'posority_thershold=40'
                                :100*np.array(porosity_batch1_sub_threshold)[4000:4100]})

measured_porosity_splitted_sorted_batch1=measured_porosity_splitted_batch1.sort_values(by=['batch1_img_1st_id','batch1_img_2nd_id']
                                                                        ).reset_index().drop(['index','batch1_img_1st_id'], axis=1)

measured_porosity_splitted_sorted_batch1.rename(columns=({'batch1_img_2nd_id':'batch_1_porosity(%)_exact'}),inplace=True)
measured_porosity_splitted_sorted_batch1['batch_1_porosity(%)_exact']=batch1['porosity(%)']
measured_porosity_splitted_sorted_batch1.head(5)
bacth2_image_id=[]
df_second_id_split=[]
df_first_id_to_integer=[]

# importing the name of images in batch2

for f in glob.glob('../input/cee-498-project8-pore-in-concrete/batch2/batch2/*.png'):
        bacth2_image_id.append(os.path.split(f)[-1])
measured_porosity=pd.DataFrame({'batch2_image_id':np.array(bacth2_image_id)})

# sorting images to compare calculated porosities in batch2 with exact porosity

df_first_id=measured_porosity['batch2_image_id'].str.split('_', expand=True)[1]
for i in df_first_id:
    df_first_id_to_integer.append(int(i))
df_second_id=measured_porosity['batch2_image_id'].str.split('_', expand=True)[2]
for i in df_second_id:
    df_second_id_split.append(int(''.join(list(filter(str.isdigit, i)))))

# representing porosity of each image at different thresholds

measured_porosity_splitted_batch2=pd.DataFrame({'batch2_img_1st_id':df_first_id_to_integer, 'batch2_img_2nd_id':df_second_id_split,'posority_thershold=0'
                                :100*np.array(porosity_batch2_sub_threshold)[0:100],'posority_thershold=9'
                                :100*np.array(porosity_batch2_sub_threshold)[900:1000],'posority_thershold=10'
                                :100*np.array(porosity_batch2_sub_threshold)[1000:1100],'posority_thershold=11'
                                :100*np.array(porosity_batch2_sub_threshold)[1100:1200],'posority_thershold=20'
                                :100*np.array(porosity_batch2_sub_threshold)[2000:2100],'posority_thershold=30'
                                :100*np.array(porosity_batch2_sub_threshold)[3000:3100],'posority_thershold=40'
                                :100*np.array(porosity_batch2_sub_threshold)[4000:4100]})

measured_porosity_splitted_sorted_batch2=measured_porosity_splitted_batch2.sort_values(by=['batch2_img_1st_id','batch2_img_2nd_id']
                                                                        ).reset_index().drop(['index','batch2_img_1st_id'], axis=1)

measured_porosity_splitted_sorted_batch2.rename(columns=({'batch2_img_2nd_id':'batch_2_porosity(%)_exact'}),inplace=True)
measured_porosity_splitted_sorted_batch2['batch_2_porosity(%)_exact']=batch2['porosity(%)']
measured_porosity_splitted_sorted_batch2.head(5)
fig, axes=plt.subplots(nrows=1, ncols=5, figsize=(30,5))

# matching calculated and exact porosities in batch1 

for i in range(5):
        ax=axes[i%5]
        ax.set_title('batch1 porosity vs. threshold= %d'%(10*(i)),fontsize=16, color='r')
        ax.scatter(measured_porosity_splitted_sorted_batch1['batch_1_porosity(%)_exact'], measured_porosity_splitted_sorted_batch1
                ['posority_thershold=%d'%(10*i)], color='blue')
        ax.set_xlabel('porosity threshold=%d'%(10*i),fontsize=16);
        plt.setp(ax.get_xticklabels(), fontsize=16);plt.setp(ax.get_yticklabels(), fontsize=16);
axes[0].set_ylabel('bacth1 exact porosity(%)',fontsize=16);

fig, axes=plt.subplots(nrows=1, ncols=5, figsize=(30,5))

# matching calculated and exact porosities in batch2 

for i in range(5):
        ax=axes[i%5]
        ax.set_title('batch2 porosity vs. threshold= %d'%(10*(i)),fontsize=16, color='r')
        ax.scatter(measured_porosity_splitted_sorted_batch2['batch_2_porosity(%)_exact'], measured_porosity_splitted_sorted_batch2
                ['posority_thershold=%d'%(10*i)], color='blue')
        ax.set_xlabel('porosity threshold=%d'%(10*i),fontsize=16);
        plt.setp(ax.get_xticklabels(), fontsize=16);plt.setp(ax.get_yticklabels(), fontsize=16);
axes[0].set_ylabel('bacth2 exact porosity(%)',fontsize=16);
fig.suptitle('test title', fontsize=20)
color = {'boxes': 'DarkGreen', 'whiskers': 'black',
               'medians': 'red', 'caps': 'Gray'}
measured_porosity_splitted_sorted_batch1.plot.box(color=color, figsize=(27,5), title='Batch1', fontsize=14).set_ylabel('batch1 porosity(%)', fontsize=14);
measured_porosity_splitted_sorted_batch2.plot.box(color=color, figsize=(27,5),title='Batch2', fontsize=14).set_ylabel('batch2 porosity(%)', fontsize=14);
mean_batch1=[];mean_batch2=[]
std_batch1=[];std_batch2=[]
cv_batch1=[];cv_batch2=[]
titles=['matching mean of 10 random images','matching std of 10 random images','matching CV of 10 random images' ]

for i in range(10):
    
#  calculating mean value of subbaches of batch 1 and 2

    mean_batch1.append(100*np.array(porosity_batch1_sub_threshold)[10*i:10*(i+1)].mean())
    mean_batch2.append(100*np.array(porosity_batch2_sub_threshold)[10*i:10*(i+1)].mean())
    
#  calculating standard deviation of subbaches of batch 1 and 2

    std_batch1.append(100*np.array(porosity_batch1_sub_threshold)[10*i:10*(i+1)].std())
    std_batch2.append(100*np.array(porosity_batch2_sub_threshold)[10*i:10*(i+1)].std())
    
#  calculating coefficient of variation of subbaches of batch 1 and 2

    cv_batch1.append(np.array(porosity_batch1_sub_threshold)[10*i:10*(i+1)].mean()/np.array(porosity_batch1_sub_threshold)[10*i:10*(i+1)].std())
    cv_batch2.append(np.array(porosity_batch2_sub_threshold)[10*i:10*(i+1)].mean()/np.array(porosity_batch1_sub_threshold)[10*i:10*(i+1)].std())
stats_batches=[[mean_batch1,mean_batch2],[std_batch1,std_batch2],[cv_batch1,cv_batch2]]
from sklearn.metrics import r2_score

fig, axes=plt.subplots(nrows=1, ncols=3, figsize=(26,6))
for i in range(3):
    ax=axes[i%3]
    ax.plot(stats_batches[i][0],stats_batches[i][1],"*", ms=8, mec="k")
    
# calculating R-squared for linear regression analysis

    z = np.polyfit(stats_batches[i][0],stats_batches[i][1], 1)
    y_hat = np.poly1d(z)(stats_batches[i][0])
    ax.plot(stats_batches[i][0], y_hat, "r-", lw=1)
    text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(stats_batches[i][1],y_hat):0.3f}$"
    ax.set_title(titles[i], fontsize=20, color='b')
    ax.set_xlabel('batch1', fontsize=16);ax.set_ylabel('batch2', fontsize=16)
    ax.legend(['matched data','linear regression'],fontsize=15)
    ax.tick_params(labelsize=14)
    ax.text(0.05, 0.65, text,transform=ax.transAxes,fontsize=15, verticalalignment='top', color='r');
import scipy
from scipy. stats import lognorm 
mu_batches=[]
sigma_batches=[]
ksquare_parameters=[]
x_lognormal=[]
y_lognormal=[]
y_lognormal_bacthes=[]
colors=['g','b']

porosity_batches=[measured_porosity_splitted_sorted_batch1['posority_thershold=0'],measured_porosity_splitted_sorted_batch2['posority_thershold=0']]
 
for i in range(2):

# fitting lognormal distribution to histograms of batches 1 and 2

# the following code is partially adapted from https://stackoverflow.com/q/26406056

    shape, location, scale = scipy.stats.lognorm.fit(porosity_batches[i])
    mu, sigma = np.log(scale), shape
    mu_batches.append(mu)
    sigma_batches.append(sigma)
    for j in range(1,1000):
        x_lognormal.append(j/10)
        y_lognormal.append(np.exp(-(np.log(j/10) - mu)**2 / (2 * sigma**2))/ (j/10* sigma * np.sqrt(2 * np.pi)))
        
#  calculating average porosity in batches 1 and 2

mu_batches_round=np.round(np.exp(mu_batches),3)    
porosity_distr=[measured_porosity_splitted_sorted_batch1['posority_thershold=0'],measured_porosity_splitted_sorted_batch2['posority_thershold=0']]
ave_porosity=[round(porosity_distr[0].mean(),3),round(porosity_distr[1].mean(),3)]


fig, axes=plt.subplots(nrows=2, ncols=2, figsize=(20,10))
for i in range(4):
        ax=axes[int(i/2)][i%2]
        ax.set_title('batch%d porosity distribution, bins=%d'%(int(i/2)+1,10**(i%2+1)),fontsize=20, color='r')
        ax.hist(porosity_distr[int(i/2)], color=colors[i%2], bins=10**(i%2+1), density=True)
        ax.plot(x_lognormal[0:999],y_lognormal[0:999], color='r')
        ax.set_xlabel('porosity(%)',fontsize=18);
        ax.set_ylabel('normalized frequency',fontsize=18);
        plt.setp(ax.get_xticklabels(), fontsize=16);plt.setp(ax.get_yticklabels(), fontsize=16);

# as well, we show the average porsoity of lognormal dist we fitted to the histograms

        ax.legend(['lognormal ave. porosity=%s'%(mu_batches_round[int(i/2)]),'batch%s ave. porosity=%s'%(int(i/2)+1,
                                                                                                           ave_porosity[int(i/2)])],fontsize=15,loc=0)
        fig.tight_layout()
import scipy.stats as stats
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
import math

# the follwoing code is partially adapted from https://stackoverflow.com/a/55732312

# the following class calculates 'ordered' medians (i.e. rank of frequency) vs response values (i.e. frequency)

class PPFScale(mscale.ScaleBase):
    name = 'ppf'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)

    def get_transform(self):
        return self.PPFTransform()

    def set_default_locators_and_formatters(self, axis):
        class VarFormatter(Formatter):
            def __call__(self, x, pos=None):
                return f'{x}'[1:]

        axis.set_major_locator(FixedLocator(np.array([.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99,.999])))
        axis.set_major_formatter(VarFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, 1e-6), min(vmax, 1-1e-6)

    class PPFTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def ___init__(self, axis, thresh):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return stats.norm.ppf(a)

        def inverted(self):
            return PPFScale.IPPFTransform()

    class IPPFTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def transform_non_affine(self, a):
            return stats.norm.cdf(a)

        def inverted(self):
            return PPFScale.PPFTransform()

mscale.register_scale(PPFScale)

if __name__ == '__main__':
   
# calclating cumulative density function (CDF)

    plt.rcParams["figure.figsize"] = (20,8)
    
#  to do probability plot analysis, our data should be sorted 

    dataSorted_batch1 = np.sort(measured_porosity_splitted_sorted_batch1['posority_thershold=0'])
    dataCdf_batch1 = np.linspace(0,1,len(dataSorted_batch1))
    dataSorted_batch2 = np.sort(measured_porosity_splitted_sorted_batch2['posority_thershold=0'])
    dataCdf_batch2 = np.linspace(0,1,len(dataSorted_batch2))

# skteching probability plots of batches 1 and 2

    plt.scatter(dataCdf_batch1, dataSorted_batch1, color='red')
    plt.scatter(dataCdf_batch2, dataSorted_batch2, color='blue')
    plt.xlabel('Log(rank of frequency)', fontsize=16)
    plt.ylabel('Log(frequency)', fontsize=16)
    plt.title('lognormal probability test of porosities in batches 1 and 2', fontsize=20, color='k')
    plt.grid(b=True, which='both')
    
# skteching linear regression (LR) of batches 1 and 2

# constants used for calculating LR of batch1

    a=dataCdf_batch1[1:100]
    b=dataSorted_batch1[1:100]

# constants used for calculating LR of batch2
    
    c=dataCdf_batch2[1:100]
    d=dataSorted_batch2[1:100]

    def powerfit(x, y, xnew):
        k, m = np.polyfit(np.log(x), np.log(y), 1)
        return np.exp(m) * xnew**(k)
    ys_batch1 = powerfit(a, b, a)
    ys_batch2 = powerfit(c, d, c)
    
    plt.plot(a,ys_batch1, color='r')
    plt.plot(c,ys_batch2, color='b')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend(['linear regression batch1','linear regression batch2','prob plot of porosities batch1','prob plot of porosities batch2'],fontsize=16)
    plt.xlim(0.01, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    correlation_matrix_batch1 = np.corrcoef(np.log(b),ys_batch1)
    correlation_xy_batch1 = correlation_matrix_batch1[0,1]
    r_squared_batch1 = correlation_xy_batch1**2
    
    correlation_matrix_batch2 = np.corrcoef(np.log(d),ys_batch2)
    correlation_xy_batch2 = correlation_matrix_batch2[0,1]
    r_squared_batch2 = correlation_xy_batch2**2

# printing R_squared values corresponds to each prob plots 

    plt.text(0.04, 0.06, 'R-squared_batch1 = %s'%(np.round(r_squared_batch1,3)), fontsize=18, color='r')
    plt.text(0.04, 3.96, 'R-squared_batch2 = %s'%(np.round(r_squared_batch2,3)), fontsize=18, color='b')
    plt.show();
fig, axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))

# comparing lognormal distributions of pores in batches 1 and 2

axes[0].plot(x_lognormal[0:999],y_lognormal[0:999], color='g');
axes[0].plot(x_lognormal[999:1998],y_lognormal[999:1998], color='b');
axes[0].axvline(7.5, color='b', linestyle='dashed',linewidth=2)
axes[0].axvline(1.5, color='g', linestyle='dashed',linewidth=2)
axes[0].legend(['bacth1','bacth2'], fontsize=14)
axes[0].tick_params(labelsize=16)
axes[0].set_xlabel('porosity', fontsize=16);axes[0].set_ylabel('normalized frequency', fontsize=16)

# matching lognormal distributions of porosities in batches 1 and 2

axes[1].plot(y_lognormal[80:998],y_lognormal[1080:1998],".", ms=8, mec="k")
z = np.polyfit(y_lognormal[80:998],y_lognormal[1080:1998], 1)
y_hat = np.poly1d(z)(y_lognormal[80:998])
axes[1].plot(y_lognormal[80:998], y_hat, "r-", lw=2)
text = f"$y={z[0]:0.3f}\;x{z[1]:+0.3f}$\n$R^2 = {r2_score(y_lognormal[1080:1998],y_hat):0.3f}$"
axes[1].text(0.05, 0.55, text,transform=axes[1].transAxes,fontsize=20, verticalalignment='top', color='r');
axes[1].tick_params(labelsize=14)
axes[1].set_xlabel('batch1 porosity lognormal fitting', fontsize=16);axes[1].set_ylabel('batch2 porosity lognormal fitting', fontsize=16);
axes[1].legend(['matched log normal distr', 'linear regression'], fontsize=16);