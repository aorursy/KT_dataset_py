import pandas as pd
def temp(v):

    try:

        return(pd.to_datetime(v))

        ####return(pd.to_datetime(v.replace('.','/').replace('//','/')))

    except:

        print(v)
pdtest=pd.read_csv('../input/test.csv') 

pdtest["Date"]=pdtest["Date"].apply(lambda v: temp(v))

pdtest["Month"]=pdtest["Date"].dt.strftime("%m")

pdtest.drop(['Customer', 'Date'], axis=1, inplace=True)

pd.set_option('display.max_row',10)

pdtest
pddtl=pd.read_csv('../input/product_details.csv', encoding='ISO-8859-1')

pddtl.rename({"Unnamed: 3" : "a"}, axis='columns', inplace=True)

pddtl.drop(['a'], axis=1, inplace=True)
pddtl.drop(['Product_Name','Pack'], axis=1, inplace=True)

pd.set_option('display.max_rows',10)

pddtl
pdstt=pd.read_csv('../input/Product_sales_train_and_test.csv') 

pdstt['Customer_BasketU']=None

for i in range(pdstt.shape[0]):

    x=pdstt['Customer_Basket'][i].split(' ')

    x=[elem.replace("\n",'') for elem in x]

    x=[elem.replace("[",'').replace("]",'') for elem in x]

    x=[elem for elem in x if(elem!='')]

    x=[elem for elem in x if(elem is not None)]

    pdstt['Customer_BasketU'][i]=x

pdstt.drop(['Customer_Basket'], axis=1, inplace=True)

pd.set_option('display.max_rows',10)

pdstt
try:

    f0=open('../Vec.csv')

except IOError:

    f0=open('../Vec.csv','w+')

    ll=list(pddtl['Product_ID'])

    for x in range(len(ll)):

        f0.write(str(ll[x]))

        f0.write(',')

    f0.write('\n')

    for i in range(pdstt.shape[0]):

        cbsk=pdstt['Customer_BasketU'][i]

        for k in range(len(ll)):

            if(str(k+1001) in cbsk):

                f0.write('1')

            else:

                f0.write('0')

            f0.write(',')

        f0.write('\n')

finally:

        f0.close()      
dtvec=pd.read_csv('../Vec.csv', encoding='ISO-8859-1')

dtvec.rename({"Unnamed: 809" : "a"}, axis='columns', inplace=True)

dtvec.drop(['a'], axis=1, inplace=True)

dtvec
pdstt.drop(['Customer_BasketU'], axis=1, inplace=True)
fpstt=pd.concat([pdstt, dtvec], axis=1)

fpstt
pdtest=pdtest.merge(fpstt, how='inner')

pdtest
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb_M = lb.fit_transform(pdtest['Month'])

lb_M_df = pd.DataFrame(lb_M, columns=lb.classes_)

pdts= pd.concat([pdtest, lb_M_df], axis=1)

pdts.drop(['Month'],axis=1, inplace=True)

#print([x for x in pdts])

pdts
Us=pdts['BillNo']

x_test=pdts.drop(['BillNo'],axis=1)

Us
pdtrain=pd.read_csv('../input/Train.csv')

pdtrain.dropna(how='all', inplace=True)

pdtrain.reset_index(drop=True,inplace=True)

pdtrain["Date"]=pdtrain["Date"].apply(lambda v: temp(v))

pdtrain["Month"]=pdtrain["Date"].dt.strftime("%m")

print(pdtrain.isnull().sum())

print(14021-pdtrain['BillNo'].nunique())

pd.set_option('display.max_row',10)

pdtrain
pml=["Discount 5%","Discount 12%","Discount 18%","Discount 28%"]

for x in pml:

    pdtrain[x].fillna(0, inplace=True)

for x in pml:

    #print(pdtrain[x].value_counts())

    print(pdtrain[x].value_counts(normalize=True)*100)

    #print(pdtrain[x].value_counts())
pdtrain['Label']='00'

lbl=['01','02','03','04','00']

for j in range(len(pml)):

    pdtrain.loc[pdtrain[pml[j]]==1, 'Label']=lbl[j]

pd.set_option('display.max_row',10)

print(pdtrain.isnull().sum())

pdtrain
d6=pdtrain.merge(fpstt, how='inner')

d6
d6.drop(['Date','Customer','Discount 5%','Discount 12%','Discount 18%','Discount 28%'], axis=1, inplace=True)
#from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

lb_M = lb.fit_transform(d6['Month'])

lb_M_df = pd.DataFrame(lb_M, columns=lb.classes_)

pdtr= pd.concat([d6, lb_M_df], axis=1)

#pdtr=d6

pdtr.drop(['Month'],axis=1, inplace=True)

pdtr
pdtr.drop(['BillNo'],axis=1, inplace=True)
c1=pdtr[pdtr['Label']=='01']

c2=pdtr[pdtr['Label']=='02']

c3=pdtr[pdtr['Label']=='03']

c4=pdtr[pdtr['Label']=='04']

c5=pdtr[pdtr['Label']=='00']

from sklearn.utils import shuffle

c1=shuffle(c1)

c2=shuffle(c2)

c3=shuffle(c3)

c4=shuffle(c4)

c5=shuffle(c5)

cnt1=int(c1.shape[0]/5)

cnt2=int(c2.shape[0]/5)

cnt3=int(c3.shape[0]/5)

cnt4=int(c4.shape[0]/5)

cnt5=int(c5.shape[0]/5)
f1=f2=f3=f4=f5=None

f1=c1[0:cnt1]

f1=f1.append(c2[0:cnt2])

f1=f1.append(c3[0:cnt3])

f1=f1.append(c4[0:cnt4])

f1=f1.append(c5[0:cnt5])

f1=shuffle(f1)

f2=c1[cnt1:2*cnt1]

f2=f2.append(c2[cnt2:2*cnt2])

f2=f2.append(c3[cnt3:2*cnt3])

f2=f2.append(c4[cnt4:2*cnt4])

f2=f2.append(c5[cnt5:2*cnt5])

f2=shuffle(f2)

f3=c1[2*cnt1:3*cnt1]

f3=f3.append(c2[2*cnt2:3*cnt2])

f3=f3.append(c3[2*cnt3:3*cnt3])

f3=f3.append(c4[2*cnt4:3*cnt4])

f3=f3.append(c5[2*cnt5:3*cnt5])

f3=shuffle(f3)

f4=c1[3*cnt1:4*cnt1]

f4=f4.append(c2[3*cnt2:int(3.9*cnt2)])

f4=f4.append(c3[3*cnt3:int(3.9*cnt3)])

f4=f4.append(c4[3*cnt4:int(3.9*cnt4)])

f4=f4.append(c5[3*cnt5:int(3.9*cnt5)])

f4=shuffle(f4)

f5=c1[int(3.9*cnt1):c1.shape[0]]

f5=f5.append(c2[int(3.95*cnt2):c2.shape[0]])

f5=f5.append(c3[int(3.95*cnt3):c3.shape[0]])

f5=f5.append(c4[int(3.95*cnt4):c4.shape[0]])

f5=f5.append(c5[int(3.95*cnt5):c5.shape[0]])

f5=shuffle(f5)
rtrn=f1.append(f2)

rtrn=rtrn.append(f3)

rtrn=rtrn.append(f4)

rtrn=shuffle(rtrn)

rtrn.reset_index(drop=True,inplace=True)

rvald=f5

rvald=shuffle(rvald)

rvald.reset_index(drop=True,inplace=True)

ttrain=rtrn.append(rvald)

ttrain=shuffle(ttrain)

ttrain.reset_index(drop=True,inplace=True)
rvald

#rtrn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn import neighbors

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
x_train, y_train=rtrn.drop(['Label'], axis=1), rtrn['Label']

x_vald, y_vald=rvald.drop(['Label'], axis=1), rvald['Label']

x_ttrn, y_ttrn=ttrain.drop(['Label'], axis=1), ttrain['Label']

xr_train,xr_vald,yr_train,yr_vald=train_test_split(x_ttrn,y_ttrn,test_size=.25)
clf=neighbors.KNeighborsClassifier(n_neighbors=9, metric='hamming' )

clf.fit(xr_train, yr_train)

print(clf)
y_predr=clf.predict(xr_vald)

print (confusion_matrix(yr_vald, y_predr))
pdt=pd.DataFrame(y_predr)

pdt1=pd.DataFrame(x_vald)

pdt=pd.concat([pdt, pdt1], axis=1)

#pdt
clf1=neighbors.KNeighborsClassifier(n_neighbors=8, metric='hamming')

clf1.fit(x_train, y_train)

print(clf1)
y_pred1=clf1.predict(x_vald)

print (confusion_matrix(y_vald, y_pred1))
clf2=neighbors.KNeighborsClassifier(n_neighbors=7)

clf2.fit(x_train, y_train)

print(clf2)
y_pred2=clf2.predict(x_vald)

print (confusion_matrix(y_vald, y_pred2))
accuracy_score(y_vald, y_pred2) #'euclidean','minkowski','hamming','canberra'
clfcrs=neighbors.KNeighborsClassifier(n_neighbors=8, metric='hamming' )

clfcrs.fit(x_train, y_train)

print(clfcrs)

y_pred2r=clfcrs.predict(x_vald)

print (confusion_matrix(y_vald, y_pred2r))
clf_crs=cross_val_score(clf, x_ttrn, y_ttrn, cv=10)

print(clf_crs)
from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier
clfl=LogisticRegression( solver= 'lbfgs',class_weight='balanced', multi_class='auto', max_iter=300)

clfl.fit(x_train, y_train)#solver='newton-cg'

print(clfl)
y_predlr=clf2.predict(x_vald)

print (confusion_matrix(y_vald, y_predlr))
clff = RandomForestClassifier(n_estimators=15 ,criterion='gini',max_features=800,

                              max_depth=100, min_samples_split=10)

clff.fit(x_train, y_train)

print(clff)
y_predrf=clff.predict(x_vald)

print (confusion_matrix(y_vald, y_predrf))
x_test.shape
x_vald.shape
y_preds=clfcrs.predict(x_test)

y_preds1=clf1.predict(x_test)
dcd={'01':"1,0,0,0",'02':"0,1,0,0",'03':"0,0,0,1",'04':"0,0,0,1",'00':"0,0,0,0"}

f6=Us

f6.reset_index(drop=True,inplace=True)

f6=pd.concat([f6,pd.DataFrame(y_preds)], axis=1)

f6

fsu=open('Submation3.csv','w+')

fsu.write("BillNo,Discount 5%,Discount 12%,Discount 18%,Discount 28%")

fsu.write('\n')

for i in range(f6.shape[0]):

    fsu.write(f6["BillNo"][i])

    fsu.write(',')

    xr=dcd[f6[0][i]]

    fsu.write(xr)

    fsu.write('\n')

fsu.close()