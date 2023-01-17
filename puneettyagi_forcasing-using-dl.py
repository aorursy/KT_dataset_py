from fastai.imports import *
from fastai.column_data import *
from fastai.structured import *
from fastai.plots import *
from fastai.sgdr import *
import pdb
PATH='../input/'
train_df=pd.read_csv(f'{PATH}train.csv',parse_dates=['date'])
test_df=pd.read_csv(f'{PATH}test.csv',parse_dates=['date'])
t_df=pd.concat([train_df.drop('sales',axis=1),test_df.drop('id',axis=1)])
add_datepart(t_df,'date')
t_df.columns
cat_col=['store', 'item', 'Year', 'Month', 'Week', 'Day', 'Dayofweek',
       'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end',
       'Is_quarter_start', 'Is_year_end', 'Is_year_start']
con_var=['Elapsed']
for c in cat_col:
    t_df[c]=t_df[c].astype('category').cat.as_ordered()
cat_sz={c:len(t_df[c].cat.categories)+1 for c in cat_col}
emb_sz=[(v,min(50,(v+1)//2)) for _,v in cat_sz.items()]
emb_sz
x_t,y,nas,mapper=proc_df(t_df,do_scale=True)
cat_sz
xtest=x_t[train_df.shape[0]:]
x=x_t[:train_df.shape[0]]
yl=np.log1p(train_df.sales)
yl.astype('float32',inplace=True)
val_idxs=np.array([a for a in range(x.shape[0]-90,x.shape[0])])
x.head()
max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
#md=ColumnarModelData.from_data_frame('',val_idxs,x,yl,cat_col,128,is_reg=True,test_df=xtest)
#learn=md.get_learner(emb_sz,1,0.04, 1, [1000,500], [0.001,0.01], y_range=y_range)
#learn.lr_find()
#def smape(f,a):
#    return (np.abs(f-a)/((np.abs(a)+np.abs(f))/2)).sum()/len(f)*100
#learn.sched.plot()
#lr=1e-5
#learn.fit(lr,1,metrics=[smape])
#learn.fit(lr,3,metrics=[smape],cycle_len=1)
#learn.fit(lr,2,metrics=[smape],cycle_len=4)
#learn.save('val')
#pred_test=learn.predict(True)
from keras.models import Sequential,Model
from keras.layers import Dense,Embedding,Activation,Input,concatenate,Flatten,merge,Reshape,BatchNormalization,Dropout
emb_layers=[]
Input_layers=[]
for input_dim,output_dim in emb_sz:
    i=Input(shape=(1,))
    Input_layers.append(i)
    i=Embedding(input_dim,output_dim)(i)
    emb_layers.append(i)
#Elapsed Time
et=Input(shape=(1,))
Input_layers.append(et)
et=Dense(1,activation='relu')(et)
et=BatchNormalization()(et)
et=Reshape((1,1))(et)

In=concatenate(emb_layers,axis=-1)
m=concatenate([In,et])
m=Dense(1000,activation='relu')(m)
m=BatchNormalization()(m)
m=Dropout(.5)(m)
m=Dense(500,activation='relu')(m)
m=BatchNormalization()(m)
m=Dropout(.5)(m)
m=Dense(1,activation='linear')(m)
model=Model(inputs=Input_layers,outputs=m)
model.compile('adam',loss='mean_absolute_error',metrics=['accuracy'])
model.summary()
xtrain,xval=x[:-90],x[-90:]
yl_tr,yl_val=yl[:-90].values.reshape(-1,1,1),yl[-90:].values.reshape(-1,1,1)
#xtrain,xval=xtrain.values.reshape(15,-1),xval.values.reshape(15,-1)
xtrain=[xtrain[col].values for col in xtrain.columns]
xval=[xval[col].values for col in xval.columns]
model.fit(x=list(xtrain), y=yl_tr, batch_size=64, epochs=1, verbose=1,validation_data=(list(xval),yl_val))
from keras.callbacks import ReduceLROnPlateau
reduceLr=ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=1,verbose=1)
model.compile('adam',loss='mean_absolute_error',metrics=['mse'])
model.fit(x=list(xtrain), y=yl_tr, batch_size=128, epochs=3, verbose=1,validation_data=(list(xval),yl_val),callbacks=[reduceLr])

model.fit(x=list(xtrain), y=yl_tr, batch_size=256, epochs=10, verbose=1,validation_data=(list(xval),yl_val),callbacks=[reduceLr])
#model.save_weights('m1.hdf5')
pred_test=model.predict([xtest[col].values for col in xtest.columns])
pred_test=np.exp(pred_test)-1
pred_test=pred_test.reshape((-1,1))
pred_test.shape
sub_df=pd.DataFrame()
sub_df['id']=test_df.id
sub_df['sales']=pred_test
sub_df.head()
sub_df.to_csv('submission.csv',index=False)
from IPython.display import FileLink
FileLink('submission.csv')
