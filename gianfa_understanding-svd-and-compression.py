import numpy as np

from numpy import linalg as LA

import matplotlib.pyplot as plt



from sklearn.datasets import fetch_olivetti_faces

from sklearn.preprocessing import StandardScaler



from ipywidgets import interact, interactive, fixed, interact_manual, Layout

import ipywidgets as widgets



from bokeh.plotting import ColumnDataSource

from bokeh.io import push_notebook, show, output_notebook

from bokeh.plotting import figure

from bokeh.layouts import widgetbox, row

output_notebook()
def plot_gallery(title, images, n_col, n_row, titles=None,cmap=plt.cm.gray):

    plt.figure(figsize=(2. * n_col, 2.26 * n_row))

    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):

        plt.subplot(n_row, n_col, i + 1)

        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape), cmap=cmap,

                   interpolation='nearest',

                   vmin=-vmax, vmax=vmax)

        plt.xticks(())

        plt.yticks(())

        if titles:

            plt.title(titles[i])

    h_s = 0.2 if titles else 0

    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, h_s)





normalize = lambda X: (X * 255/X.max()).astype(int)    



# Parameters

image_shape = (64, 64)

rng = np.random.seed(0)
# Load 1 face from olivetti dataset

dataset = fetch_olivetti_faces(shuffle=True, random_state=rng).data #  we random extract one face

data = dataset[100].reshape(image_shape) # the 101th face



print("Image (dataset) shape: %s" % str(data.shape))
pic = data.reshape(image_shape)

plt.imshow(pic)

plt.title('Original image');
scaler = StandardScaler()      # 

X = normalize(data)#scaler.fit_transform(data) # 1. standardize data

U, S_vec, Vt = LA.svd(X)       # 2. apply svd
### Visualize

print('U:',U.shape, ', S vector: ', S_vec.shape, ', V*:', Vt.shape)

# Build the singular values matrix since numpy has returned a vector 

S = np.zeros( X.shape, dtype=complex)

S[:len(S), :len(S)] = np.diag(S_vec)

# Reconstruct the image normalizing values on 256 levels

X_r = normalize( np.dot(U, np.dot(S, Vt)) )

X_r
plt.imshow( X_r )

plt.title('Reconstructed image');
X.max()
plt.figure(figsize=(15,15))

plt.subplot(1,3,1)

plt.imshow( pic )

plt.title('Original image')

plt.subplot(1,3,2)

plt.imshow( X )

plt.title('Original Standardized')

plt.subplot(1,3,3)

plt.imshow( X_r )

plt.title('Reconstructed image');

print( 'MAE: ',  np.sum(np.abs(X - X_r)) / np.prod(X.shape) )
# Now truncate the matrices

upto = 9 # number of singular values to consider, previously:image_shape[0]

S_tr = S.copy()

S_tr[upto:,upto:] = 0  # set to zero all singular values over the 'upto'th

X_r = np.dot(U, np.dot(S_tr, Vt))

X_r = X_r * 255/X_r.max()



plt.imshow( X_r.astype(int) )

plt.title('Reconstructed image');
plt.figure(figsize=(15,15))

plt.subplot(1,3,1)

plt.imshow( U )

plt.title('U, left vectors')

plt.subplot(1,3,2)

plt.imshow( S.astype(int), cmap='Greys' )

plt.title('S, singular values')

plt.subplot(1,3,3)

plt.imshow( Vt )

plt.title('Vt, conj transpose of right vectors');
# Increasing number of non zero singular values

projs  = [] 

proj_t = []

upto = 9 # number of singular values to consider, previously:image_shape[0]

for c in range(upto):

    #c = 1

    S_tr = S.copy()

    S_tr[c:, c:] = 0

    X_n_x = np.dot(U, np.dot(S_tr, Vt))

    X_n_x = (X_n_x * 255/X_n_x.max()).astype(int)

    projs.append(X_n_x)

    proj_t.append(c)



n_col, n_row = 3, 3 

plot_gallery('', projs, n_col, n_row, titles=proj_t)

fig = plt.gcf()

fig.suptitle("Increasing number of non zero singular values representations", fontsize=14, y=1);
upto = 6

U_tr = U[:, :upto]

S_tr = S[:upto, :upto]

Vt_tr = Vt[:upto, :]

X_tr_n = normalize( np.dot(U_tr, np.dot( S_tr, Vt_tr )) )

plt.imshow(X_tr_n);







print(

'Dimension comparison\n',

    'U: ',U.shape, '=', np.prod(U.shape) , '; U_tr: ',U_tr.shape, '=', np.prod(U_tr.shape) , '\n',

    'S: ',S.shape, '=', np.prod(S.shape) , '; S_tr: ',S_tr.shape, '=', np.prod(S_tr.shape) , '\n',

    'V: ',Vt.shape, '=', np.prod(Vt.shape) , '; V_tr: ',Vt_tr.shape, '=', np.prod(Vt_tr.shape) , '\n',

    'X: ',X.shape, '=', np.prod(X.shape) , '; X_tr: ',X_tr_n.shape, '=', np.prod(X_tr_n.shape) , '\n',    

    '________________', '\n',

    'U+S+V = ', str(np.prod(U.shape)+np.prod(S.shape)+np.prod(Vt.shape)), '\n',

    'U_tr+S_tr+V_tr = ', str(np.prod(U_tr.shape)+np.prod(S_tr.shape)+np.prod(Vt_tr.shape)),'\n',

    'Compressed to', str(

        (np.prod(U_tr.shape)+np.prod(S_tr.shape)+np.prod(Vt_tr.shape))*100 / (np.prod(U.shape)+np.prod(S.shape)+np.prod(Vt.shape))

        

        ),'%'

    )
# first nine singular values single combination

projs  = [] 

proj_t = []

n_comp = 9

for c in range(n_comp):

    #c = 1

    S_x = np.full_like(S, 0)

    S_x[c, c] = S[c, c]

    X_n_c = np.dot(U, np.dot(S_x, Vt))

    X_n_c = (X_n_c * 255/X_n_c.max()).astype(int)

    projs.append(X_n_c)

    proj_t.append(c)



n_col, n_row = 3, 3 

plot_gallery('', projs, n_col, n_row, titles=proj_t)

fig = plt.gcf()

fig.suptitle("Single first singular values combinations", fontsize=14, y=1);
a,b,c,d,e,f,g,h,i = projs # first 9 components
k = np.array([    # coefficients of the first 9 elements

    1, 0.3, 0.8, 1,   

    0, 0, 0, 0,

    0,

    ])

#k = k/k.sum()

# todo: normalize X_n_c, proj = proj/proj.sum

som = k[0]*a + k[1]*b + k[2]*c + k[3]*d + k[4]*e + k[5]*f + k[6]*g + k[7]*h + k[8]*i

somn = normalize(som)

plt.imshow(somn)
plt.figure(figsize=(14,4))

plt.subplot(2,4,1)

plt.imshow(k[0]*a)

plt.title( 'A = ' + str(round(k[0],2)) + 'a' )



plt.subplot(2,4,2)

plt.imshow(k[1]*b)

plt.title( 'B = ' + str(round(k[0],2)) + 'b')



plt.subplot(2,4,3)

plt.imshow(k[2]*c)

plt.title( 'C = ' + str(round(k[1],2)) + 'c')



plt.subplot(2,4,4)

plt.imshow(k[3]*d)

plt.title( 'D = ' + str(round(k[2],2)) + 'd')



plt.subplot(2,4,5)

plt.imshow(k[4]*e)

plt.title( 'E = ' + str(round(k[3],2)) + 'e')



plt.imshow( normalize(k[0]*a + k[1]*b) )

plt.title('A+B')

plt.subplot(2,4,6)

plt.imshow( normalize(k[0]*a + k[1]*b + k[2]*c ))

plt.title('A+B+C')

plt.subplot(2,4,7)

plt.imshow( normalize(k[0]*a + k[1]*b + k[2]*c + k[3]*d ))

plt.title('A+B+C+D');

plt.subplot(2,4,8)

plt.imshow( normalize(k[0]*a + k[1]*b + k[2]*c + k[3]*d + k[4]*e ))

plt.title('A+B+C+D+E');

plt.subplots_adjust(hspace=0.45)

fig = plt.gcf()

fig.suptitle("Single singular value contribution vs Cumulative result", fontsize=14, y=1);
sv = 3 # number of the singular value

S_x = np.full_like(S, 0)

S_x[sv, sv] = S[sv, sv]

US = np.dot(U, S_x)

USVt = np.dot(US, Vt)







plt.figure(figsize=(14,14))

plt.subplot(1,3,1)

plt.imshow(normalize(US))

plt.title('$US_c$')

plt.subplot(1,3,2)

plt.imshow(Vt)

plt.title('$V\ast$')

plt.subplot(1,3,3)

plt.imshow(normalize(USVt))

plt.title('$US_c V^{/ast}$');

# plt.subplot(1,3,3)

# plt.imshow(normalize(np.dot(U,np.dot(S_x, Vt))))

# plt.title('$USV^{/ast}$')
S_x = np.full_like(S, 0)

S_x[sv, sv] = S[sv, sv]



SVt   = np.dot(S_x, Vt)

USVt2 = np.dot(U, SVt)



plt.figure(figsize=(14,14))

plt.subplot(1,3,1)

plt.imshow(normalize(SVt))

plt.title('$S_c V^{*}$')

plt.subplot(1,3,2)

plt.imshow(normalize(USVt2))

plt.title('$US_c V^{*}$');
S_zeros = np.full_like(S, 0)

def get_prods_by_c(c):

    S_x = S_zeros.copy()

    S_x[c, c] = S[c, c]

    US_c = np.dot(U, S_x)

    USVt_c = np.dot(US_c, Vt)

    p1.image(image=[normalize(US_c)], x=0, y=0, dw=1, dh=1)

    p2.image(image=[normalize(S_x)], x=0, y=0, dw=1, dh=1)

    p3.image(image=[normalize(USVt_c)], x=0, y=0, dw=1, dh=1)

    push_notebook()





import numpy as np

from bokeh.plotting import figure, show



pw = 300

ph = 300

p1 = figure(x_range=(0, 1), y_range=(0, 1), plot_width=pw , plot_height=ph, title='US')

p2 = figure(x_range=(0, 1), y_range=(0, 1), plot_width=pw , plot_height=ph, title='S')

p3 = figure(x_range=(0, 1), y_range=(0, 1), plot_width=pw , plot_height=ph, title='USV*')

# must give a vector of image data for image parameter

p1.image(image=[normalize(S)], x=0, y=0, dw=1, dh=1)

p2.image(image=[normalize(S)], x=0, y=0, dw=1, dh=1)

p3.image(image=[normalize(S)], x=0, y=0, dw=1, dh=1)

# show(p)



layout = row([p2, p1, p3])

interact(get_prods_by_c, c=widgets.IntSlider(min=0, max=20, continous_update=False, layout=Layout(width='500px')))       

show(layout, notebook_handle=True);