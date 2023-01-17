import pandas as pd, numpy as np, os

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
PATH = '../input/melanoma-oof-and-sub/'

FILES = os.listdir(PATH)



OOF = np.sort( [f for f in FILES if 'oof' in f] )

OOF_CSV = [pd.read_csv(PATH+k) for k in OOF]



print('We have %i oof files...'%len(OOF))

print(); print(OOF)
x = np.zeros(( len(OOF_CSV[0]),len(OOF) ))

for k in range(len(OOF)):

    x[:,k] = OOF_CSV[k].pred.values

    

TRUE = OOF_CSV[0].target.values
all = []

for k in range(x.shape[1]):

    auc = roc_auc_score(OOF_CSV[0].target,x[:,k])

    all.append(auc)

    print('Model %i has OOF AUC = %.4f'%(k,auc))

    

m = [np.argmax(all)]; w = []
old = np.max(all); 



RES = 200; 

PATIENCE = 10; 

TOL = 0.0003

DUPLICATES = False



print('Ensemble AUC = %.4f by beginning with model %i'%(old,m[0]))

print()



for kk in range(len(OOF)):

    

    # BUILD CURRENT ENSEMBLE

    md = x[:,m[0]]

    for i,k in enumerate(m[1:]):

        md = w[i]*x[:,k] + (1-w[i])*md

        

    # FIND MODEL TO ADD

    mx = 0; mx_k = 0; mx_w = 0

    print('Searching for best model to add... ')

    

    # TRY ADDING EACH MODEL

    for k in range(x.shape[1]):

        print(k,', ',end='')

        if not DUPLICATES and (k in m): continue

            

        # EVALUATE ADDING MODEL K WITH WEIGHTS W

        bst_j = 0; bst = 0; ct = 0

        for j in range(RES):

            tmp = j/RES*x[:,k] + (1-j/RES)*md

            auc = roc_auc_score(TRUE,tmp)

            if auc>bst:

                bst = auc

                bst_j = j/RES

            else: ct += 1

            if ct>PATIENCE: break

        if bst>mx:

            mx = bst

            mx_k = k

            mx_w = bst_j

            

    # STOP IF INCREASE IS LESS THAN TOL

    inc = mx-old

    if inc<=TOL: 

        print(); print('No increase. Stopping.')

        break

        

    # DISPLAY RESULTS

    print(); #print(kk,mx,mx_k,mx_w,'%.5f'%inc)

    print('Ensemble AUC = %.4f after adding model %i with weight %.3f. Increase of %.4f'%(mx,mx_k,mx_w,inc))

    print()

    

    old = mx; m.append(mx_k); w.append(mx_w)
print('We are using models',m)

print('with weights',w)

print('and achieve ensemble AUC = %.4f'%old)
md = x[:,m[0]]

for i,k in enumerate(m[1:]):

    md = w[i]*x[:,k] + (1-w[i])*md

plt.hist(md,bins=100)

plt.title('Ensemble OOF predictions')

plt.show()
df = OOF_CSV[0].copy()

df.pred = md

df.to_csv('ensemble_oof.csv',index=False)
SUB = np.sort( [f for f in FILES if 'sub' in f] )

SUB_CSV = [pd.read_csv(PATH+k) for k in SUB]



print('We have %i submission files...'%len(SUB))

print(); print(SUB)
# VERFIY THAT SUBMISSION FILES MATCH OOF FILES

a = np.array( [ int( x.split('_')[1].split('.')[0]) for x in SUB ] )

b = np.array( [ int( x.split('_')[1].split('.')[0]) for x in OOF ] )

if len(a)!=len(b):

    print('ERROR submission files dont match oof files')

else:

    for k in range(len(a)):

        if a[k]!=b[k]: print('ERROR submission files dont match oof files')
y = np.zeros(( len(SUB_CSV[0]),len(SUB) ))

for k in range(len(SUB)):

    y[:,k] = SUB_CSV[k].target.values
md2 = y[:,m[0]]

for i,k in enumerate(m[1:]):

    md2 = w[i]*y[:,k] + (1-w[i])*md2

plt.hist(md2,bins=100)

plt.show()
df = SUB_CSV[0].copy()

df.target = md2

df.to_csv('ensemble_sub.csv',index=False)