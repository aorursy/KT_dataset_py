import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tqdm import tqdm

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def hist2D(x,y,bins,r,c,norm=False,gain=1):

    h = np.histogram2d(x,y,bins,range=r)[0].T.reshape((bins,bins,1))

    H = np.concatenate((h*c[0],h*c[1],h*c[2]),axis=2)

    

    if norm:

        H = (H/H.max())**(1/gain)

        H[H>1]=1

        return H

    return H



def histplot(x, 

             y, 

             labels, 

             bins=100, 

             normalize_each_label=False, 

             range=None, 

             colors=[[1,0,0],[0,1,0]],

             gain = 1):

    

    # If Range is not specified set it to the min and max of x and y respectively

    if range==None:

        range=np.array(((x.min(),x.max()),(y.min(),y.max())))

    else:

        range = np.array(range)

        if not range.shape == (2,2):

            raise ValueError('range should be array-like with shape (2,2). {} is not valid'.format(range))

    

    

    # Initiallize RGB image to zeros

    H = np.zeros((bins,bins,3))

    

    

    # Add each label's histogram to H

    for i,l in enumerate(list(set(labels))):

        idxs = np.where(labels==l)[0]

        H = H + hist2D(x[idxs],

                       y[idxs],

                       bins,

                       range,

                       colors[l],

                       normalize_each_label,

                       gain

                      )

    

    # Normalize and apply gain

    im = H/H.max()

    im[im.sum(2)==0]=1

    

    # Plot image

    plt.imshow(np.flip(im,0)) # Must be flipped because vertex is at top left for images

    

    

    # Draw axes

    range_x = np.round(np.linspace(range[0][0],range[0][1],bins),2)

    range_y = np.round(np.linspace(range[1][0],range[1][1],bins),2)

    _ = np.arange(0,bins-1,(bins-1)//5)

    plt.xticks(_, range_x[_])

    plt.yticks(_, range_y[-_-1])

    

    # Show image

    plt.show()

    
df = pd.read_csv('../input/train.csv')
df.head()
print(df.loc[:10,'wheezy-copper-turtle-magic'].tolist())
w = df.loc[:,'wheezy-copper-turtle-magic']

print(w.min(),w.max())
column_names = df.columns[1:-1]
n=0

m=1



x = df[column_names[n]].values

y = df[column_names[m]].values

t = df['target'].values





histplot(

    x,

    y,

    t,

    bins=500,

#     range=((-3,3.2),(-3,3)),

    normalize_each_label=True,

    colors = [

        [1,0,0],

        [0,1,1]],

    gain=20)
histplot(

    x,

    y,

    t,

    bins=500,

    range=((-3,3.2),(-3,3)),

    normalize_each_label=True,

    colors = [

        [1,0,0],

        [0,1,1]],

    gain=20)
cond = df['wheezy-copper-turtle-magic'].values

q = 0





new_x = x[np.where(cond==q)[0]]

new_y = y[np.where(cond==q)[0]]

new_t = t[np.where(cond==q)[0]]



histplot(

    x,

    y,

    t,

    bins=500,

#     range=((-3,3.2),(-3,3)),

    normalize_each_label=True,

    colors = [

        [1,0,0],

        [0,1,1]],

    gain=20)





for i in range(20):

    q = i





    new_x = x[np.where(cond==q)[0]]

    new_y = y[np.where(cond==q)[0]]

    new_t = t[np.where(cond==q)[0]]

    

    histplot(

        new_x,

        new_y,

        new_t,

        bins=100,

        range=((-15,15),(-15,15)),

        normalize_each_label=False,

        colors = [

            [1,0,0],

            [0,1,0]],

        gain=2)
print(cond.max())
STDs = []



for j in column_names:

    std = []

    x = df[j].values

    for i in range(512):

        x_q = x[np.where(cond==i)[0]]

        std.append(x_q.std())

    STDs.append(std)



STDs = np.array(STDs)

plt.imshow(STDs)
plt.hist(STDs.reshape(-1,),bins =100)

# plt.ylim(0,40)

plt.title('Standard Deviation for a given Variable and Wheezy value')

plt.xlabel('Standard Deviation')

plt.ylabel('# of combinations with std=x')

plt.show()

import scipy.stats as stats



peak_1 = STDs.reshape(-1,)[np.where(STDs.reshape(-1,)<2)[0]]

peak_1 = peak_1[np.where(peak_1>0)[0]]

peak_2 = STDs.reshape(-1,)[np.where(STDs.reshape(-1,)>2)[0]]







h,bins = np.histogram(peak_1,bins =500)

h = h/h.max()

plt.plot(bins[1:],h)



plt.title('Peak 1')

plt.xlabel('Standard Deviation')

plt.ylabel('# of combinations with std=x')



x = np.linspace(0.9,1.1,500)

y = stats.norm.pdf(x, peak_1.mean(), peak_1.std())

y = y/y.max()



plt.plot(x,y,color='orange')

plt.show()





h,bins = np.histogram(peak_2,bins =500)

h = h/h.max()

plt.plot(bins[1:],h)



plt.title('Peak 2')

plt.xlabel('Standard Deviation')

plt.ylabel('# of combinations with std=x')



x = np.linspace(2.9,4.6,500)

y = stats.norm.pdf(x, peak_2.mean(), peak_2.std())

y = y/y.max()



plt.plot(x,y,color='orange')

plt.show()
m=4

idxs_w_0 = np.where(cond==m)[0]

x_w_0 = df.loc[idxs_w_0,column_names[STDs[:,m]>2]].values

y_w_0 = df.loc[idxs_w_0,'target'].values
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import normalize

from sklearn.svm import SVC



x_w_0 = normalize(x_w_0, norm='max', axis=0, copy=False, return_norm=False)

x_train, x_test, y_train, y_test = train_test_split(x_w_0, y_w_0, test_size=0.1,random_state=42)



model = SVC(C=5.0, 

            kernel='rbf',

            gamma='scale',

            shrinking=True, 

            probability=False, 

            tol=0.001, 

            cache_size=200,

            max_iter=-1, 

            decision_function_shape='ovr')





model.fit(x_train,y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

print(acc)



models = {}

accs = []

for i in tqdm(range(512)):

    idxs = np.where(cond==i)[0]

    c = column_names[STDs[:,i]>2]

    

    x_w = df.loc[idxs, c].values

    y = df.loc[idxs,'target'].values

    

    x, s = normalize(x_w, norm='max', axis=0, copy=False, return_norm=True)

    

    model = SVC(C=2, 

            kernel='rbf',

            gamma='scale',

            shrinking=False, 

            probability=False)



    model.fit(x,y)

    models[i] = [model,s,c]

    
test_df = pd.read_csv('../input/test.csv')
ids = []

predictions = []

for i in tqdm(range(512)):

    x_df = test_df.loc[test_df['wheezy-copper-turtle-magic'] == i]

    ids = ids + x_df.loc[:,'id'].values.tolist()

    m,s,c = models[i]

    x = x_df.loc[:,c].values

    x = np.divide(x,s)

    p = m.predict(x).tolist()

    predictions = predictions+p
print(len(ids),len(predictions))
df = pd.DataFrame({'id': ids,

                   'target': predictions})

df.to_csv('submission.csv',index=False)
