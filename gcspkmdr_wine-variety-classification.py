# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from fastai.text import *
from fastai import *

sns.set_style('darkgrid')
%matplotlib inline

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


fnames=['/kaggle/input/pretrained-models/lstm_fwd','/kaggle/input/pretrained-models/itos_wt103']
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/wine-dataset/train.csv')
test_data = pd.read_csv('/kaggle/input/wine-dataset/test.csv')
train_data.shape
train_data.head()
def nullColumns(train_data):
    list_of_nullcolumns =[]
    for column in train_data.columns:
        total= train_data[column].isna().sum()
        if total !=0:
            print('Total Na values is {0} for column {1}' .format(total, column))
            list_of_nullcolumns.append(column)
    print('\n')
    return list_of_nullcolumns


def percentMissingFeature(data):
    data_na = (data.isnull().sum() / len(data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :data_na})
    print(missing_data.head(20))
    return data_na


def plotMissingFeature(data_na):
    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='90')
    if(data_na.empty ==False):
        sns.barplot(x=data_na.index, y=data_na)
        plt.xlabel('Features', fontsize=15)
        plt.ylabel('Percent of missing values', fontsize=15)
        plt.title('Percent missing data by feature', fontsize=15)
train_data_na = percentMissingFeature(train_data)
nullColumns(train_data)
plotMissingFeature(train_data_na)
test_data_na = percentMissingFeature(test_data)
nullColumns(test_data)
plotMissingFeature(test_data_na)
#train_data =  train_data.drop(columns = ['region_2','designation','user_name','price','points'],axis =1)
#test_data =  test_data.drop(columns = ['region_2','designation','user_name','price','points'],axis =1)
train_data['country'] = train_data['country'].fillna('unknown')
train_data['region_1'] = train_data['region_1'].fillna('unknown')
train_data['province'] = train_data['province'].fillna('unknown')
train_data['price'] =  train_data['price'].fillna(train_data['price'].mean())
train_data['quality/price']  =np.array(np.log1p(train_data['points']))/np.array(np.log1p(train_data['price']))
train_data['value_for_money'] = train_data['quality/price'].apply(lambda val : 'High' if val > 1.5 else ('Medium' if val > 1.0 else 'Low'))
def redness_score(descriptions):
    return (descriptions.count('cherry',re.I) +
           descriptions.count('berry',re.I) +
           descriptions.count('cherries',re.I) +
           descriptions.count('berries',re.I) +
           descriptions.count('red',re.I) +
           descriptions.count('raspberry',re.I) +
           descriptions.count('raspberries',re.I) +
           descriptions.count('blueberry',re.I) +
           descriptions.count('blueberries',re.I)+
           descriptions.count('blackberry',re.I)+
           descriptions.count('blackberries',re.I))


    
def whiteness_score(descriptions):
    return (descriptions.count("lemon",re.I)+
            descriptions.count("lemons",re.I)+
            descriptions.count("lime",re.I)+
            descriptions.count("limes",re.I)+
            descriptions.count("peach",re.I)+
            descriptions.count("peaches",re.I)+
            descriptions.count("white",re.I)+
            descriptions.count("apricot",re.I)+
            descriptions.count("pear",re.I)+
            descriptions.count("apple",re.I)+
            descriptions.count("nectarine",re.I)+
            descriptions.count("orange",re.I)+
            descriptions.count("pineapple",re.I))
        
train_data['redness_score'] = train_data['review_description'].apply(lambda r: redness_score(r))
train_data['whiteness_score'] = train_data['review_description'].apply(lambda w: whiteness_score(w))
wrs = train_data.groupby('variety')['redness_score'].describe().sort_values(by = ['mean'], ascending = False)
f,ax = plt.subplots(figsize = (10,10))
sns.barplot(wrs['mean'],wrs.index)
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlabel("Average Score",fontsize =20)
plt.ylabel("Variety",fontsize =20,rotation = 0)
plt.title('Wines Redness Score', fontsize=25)
plt.grid(True)
plt.show()
wws = train_data.groupby('variety')['whiteness_score'].describe().sort_values(by = ['mean'], ascending = False)
f,ax = plt.subplots(figsize = (10,10))
sns.barplot(wws['mean'],wws.index)
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlabel("Average Score",fontsize =20)
plt.ylabel("Variety",fontsize =20,rotation = 0)
plt.title('Wines Whiteness Score', fontsize=25)
plt.grid(True)
plt.show()
cwv = train_data.groupby('variety')['price'].describe().sort_values(by = ['mean'], ascending = False)
f,ax = plt.subplots(figsize = (10,10))
sns.barplot(cwv['mean'],cwv.index)
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlabel("Average Price",fontsize =20)
plt.ylabel("Variety",fontsize =20,rotation = 0)
plt.title('Wines Average Cost', fontsize=25)
plt.grid(True)
plt.show()
hrw = train_data.groupby('country')['points'].describe().sort_values(by = ['mean'], ascending = False)[0:5]
f,ax = plt.subplots(figsize = (10,10))
sns.pointplot(hrw.index,hrw['mean'],sort = False)
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlabel("Country",fontsize =20)
plt.ylabel("Average Points",fontsize =20,rotation = 0)
plt.title('Top 5 Highly Rated Wine Producing countries', fontsize=25)
plt.grid(True)
plt.show()
cw = train_data.groupby('country')['price'].describe().sort_values(by = ['mean'], ascending = False)[0:5]
f,ax = plt.subplots(figsize = (10,10))
sns.pointplot(cw.index,cw['mean'],sort = False)
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlabel("Country",fontsize =20)
plt.ylabel("Average Price",fontsize =20,rotation = 0)
plt.title('Top 5 Costliest Wine Producing countries', fontsize=25)
plt.grid(True)
plt.show()
vfm = train_data.groupby('country')['quality/price'].describe().sort_values(by = ['mean'], ascending = False)[0:10]
f,ax = plt.subplots(figsize = (10,10))
sns.pointplot(vfm.index,vfm['mean'],sort = False,)
plt.xticks(fontsize =15,rotation =90)
plt.yticks(fontsize =15)
plt.xlabel("Country",fontsize =20)
plt.ylabel("Value for Money ",fontsize =20)
plt.title('Top 10 Value for Money Wine Producing countries', fontsize=25)
plt.grid(True)
plt.show()
cw = train_data.groupby(by = ['country','winery'])['price'].describe().sort_values(by = ['mean'], ascending = False)
ccw =cw[0:20]
f,ax = plt.subplots(figsize = (20,20))
sns.barplot(ccw.index,ccw['mean'])
plt.xticks(fontsize =15,rotation =90)
plt.yticks(fontsize =15)
plt.xlabel("(Country,Winery)",fontsize =20)
plt.ylabel("Price ",fontsize =20)
plt.title('Costliest Wineries', fontsize=25)
plt.show()
ecw = cw[-20:]
f,ax = plt.subplots(figsize = (20,20))
sns.barplot(ecw.index,ecw['mean'])
plt.xticks(fontsize =15,rotation =90)
plt.yticks(fontsize =15)
plt.xlabel("(Country,Winery)",fontsize =20)
plt.ylabel("Price ",fontsize =20)
plt.title('Economic Wineries', fontsize=25)
plt.show()
hrv = train_data.groupby('variety')['points'].describe().sort_values(by = ['mean'], ascending = False)[0:5]
f,ax = plt.subplots(figsize = (10,10))
sns.pointplot(hrv.index,hrv['mean'],sort = False)
plt.xticks(fontsize =15,rotation =90)
plt.yticks(fontsize =15)
plt.xlabel("Wines",fontsize =20)
plt.ylabel("Average Points",fontsize =20,rotation = 0)
plt.title('Top 5 Highly Rated Wines', fontsize=25)
plt.grid(True)
plt.show()
f, axs = plt.subplots(1, 3, figsize=(40, 40))
for ax, wine in zip(axs, hrv.index[0:3]):
    review = " ".join(text for text in train_data[train_data['variety'] == wine]['review_description'])
    stopwords = set(STOPWORDS)
    # upadating stop words to eliminate name of varieties from review description
    stopwords.update(["drink", "now", "wine", "flavor", "flavors","palate",'Nebbiolo', 
                      'Grüner', 'Veltliner', 'Blend','Champagne', 'Riesling','Pinot','Noir'])

    # Generate a word cloud image
    wcl = WordCloud(stopwords=stopwords, background_color="black").generate(review)

    ax.imshow(wcl, interpolation='bilinear',aspect = 'equal')
    ax.set_title(wine.capitalize())
    ax.grid(False)

plt.show()
train_data['all_text_combined0'] = train_data['review_title'] +" " + train_data['review_description'] #for training classifier model 2
train_data['all_text_combined1'] = train_data['country'] +" " + train_data['review_title'] +" " + train_data['review_description'] #for training classifier model 3
train_data['all_text_combined2'] = train_data['country'] +" " + train_data['province'] +" " + train_data['review_title'] +" " + train_data['review_description'] #for training classifier model 4
train_data['all_text_combined3'] = train_data['country'] +" " + train_data['province'] +" " + train_data['region_1'] +" " + train_data['review_title'] +" " + train_data['review_description'] #for training classifier model 5
train_data['all_text_combined4'] = train_data['value_for_money'] +" " + train_data['review_title'] +" " + train_data['review_description'] #for training classifier model 6
test_data['country'] = test_data['country'].fillna('unknown')
test_data['region_1'] = test_data['region_1'].fillna('unknown')
test_data['province'] = test_data['province'].fillna('unknown')
test_data['price'] =  test_data['price'].fillna(test_data['price'].mean())
test_data['quality/price']  =np.array(np.log1p(test_data['points']))/np.array(np.log1p(test_data['price']))
test_data['value_for_money'] = test_data['quality/price'].apply(lambda val : 'High' if val > 1.5 else ('Medium' if val > 1.0 else 'Low'))
test_data['all_text_combined0'] = test_data['review_title'] +" " + test_data['review_description']#for testing classifier model 2
test_data['all_text_combined1'] = test_data['country'] +" " + test_data['review_title'] +" " + test_data['review_description'] #for testing classifier model 3
test_data['all_text_combined2'] = test_data['country'] +" " + test_data['province'] +" " + test_data['review_title'] +" " + test_data['review_description'] #for testing classifier model 4
test_data['all_text_combined3'] = test_data['country'] +" " + test_data['province'] +" " + test_data['region_1'] +" " + test_data['review_title'] +" " + test_data['review_description'] #for testing classifier model 5
test_data['all_text_combined4'] = test_data['value_for_money'] +" " + test_data['review_title'] +" " + test_data['review_description'] #for training classifier model 6
train_data.head(5)
train_data_lm1 = (TextList.from_df(df=train_data,path='.',cols='review_description').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
train_data_lm1.save('train_data_lm1.pkl')
train_data_lm1.vocab.itos[:10]
train_data_lm1.train_ds[0][0]
train_data_lm1 = load_data('', 'train_data_lm1.pkl', bs=48)
train_data_lm1.show_batch(5)
languageModel1 = language_model_learner(train_data_lm1, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel1.lr_find()
languageModel1.recorder.plot(skip_end=10)
languageModel1.fit_one_cycle(5, 1e-2)
languageModel1.save('fine_tuned1')
languageModel1.save_encoder('fine_tuned_enc1')
data_classifier1 = (TextList.from_df(df=train_data,path='.',cols='review_description', vocab=train_data_lm1.vocab)
                     .split_by_rand_pct(0.1)
                     .label_from_df('variety')
                     .add_test(test_data)
                     .databunch(bs=48))
data_classifier1.save('data_classifier1.pkl')
data_classifier1 = load_data('','data_classifier1.pkl',bs=48)
data_classifier1.show_batch(10)
classifierModel1 = text_classifier_learner(data_classifier1, arch=AWD_LSTM, drop_mult=0.5)
classifierModel1.load_encoder('fine_tuned_enc1')
classifierModel1.freeze()
classifierModel1.summary()
classifierModel1.lr_find()
classifierModel1.recorder.plot()
classifierModel1.fit_one_cycle(1, 2e-2)
for i in range(2,5):
    classifierModel1.freeze_to(-i)
    classifierModel1.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    print ('')
classifierModel1.recorder.plot_losses() ##still have to figure out why validation losses are not getting plotted
classifierModel1.unfreeze()
classifierModel1.fit_one_cycle(2, slice(1e-5/(2.6**4),1e-3))
classifierModel1.fit_one_cycle(5, slice(1e-5,1e-3))
print ('')
classifierModel1.save('classifierModel1')
classifierModel1.show_results(rows = 100)
train_loss1,train_accuracy1 = classifierModel1.validate(classifierModel1.data.train_dl)
valid_loss1,valid_accuracy1 = classifierModel1.validate(classifierModel1.data.train_dl)
print("Training Accuracy from model 1 is {0}".format(train_accuracy1*100))
print('')
print("Validation Accuracy from model 1 is {0}".format(valid_accuracy1*100))
def build_submission_file(model,predictions,fileName):
    
    labels = []
    final_data =pd.read_csv('/kaggle/input/wine-dataset/test.csv')
    
    for i in predictions[0]:
        labels.append(model.data.classes[np.argmax(i)])
    
    submission_final = pd.concat([final_data,pd.DataFrame(labels,columns = ['variety'])],
                     axis = 1)
    submission_final.to_csv(fileName,index = False, header=True)
    return submission_final

    
    
preds1 = classifierModel1.get_preds(DatasetType.Test)
build_submission_file(classifierModel1,preds1,'submission1.csv')
train_data_lm2 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined0').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
train_data_lm2.save('train_data_lm2.pkl')
train_data_lm2.vocab.itos[:10]
train_data_lm2.train_ds[0][0]
train_data_lm2 = load_data('', 'train_data_lm2.pkl', bs=48)
train_data_lm2.show_batch(5)
languageModel2 = language_model_learner(train_data_lm2, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel2.lr_find()
languageModel2.recorder.plot(skip_end=10)
languageModel2.fit_one_cycle(5, 1e-2)
#languageModel2.save('fine_tuned2')
languageModel2.save_encoder('fine_tuned_enc2')
data_classifier2 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined0', vocab=train_data_lm2.vocab)
                    .split_by_rand_pct(0.1)
                    .label_from_df('variety')
                    .add_test(test_data)
                    .databunch(bs=48))
data_classifier2.save('data_classifier2.pkl')
data_classifier2 = load_data('','data_classifier2.pkl',bs=48)
data_classifier2.show_batch(5)
classifierModel2 = text_classifier_learner(data_classifier2, arch=AWD_LSTM, drop_mult=0.5)
classifierModel2.load_encoder('fine_tuned_enc2')
classifierModel2.freeze()
classifierModel2.summary()
classifierModel2.lr_find()
classifierModel2.recorder.plot()
classifierModel2.fit_one_cycle(1, 2e-2)
for i in range(2,5):
    classifierModel2.freeze_to(-i)
    classifierModel2.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    print ('')
classifierModel2.recorder.plot_losses()
train_loss2,train_accuracy2 = classifierModel2.validate(classifierModel2.data.train_dl)
valid_loss2,valid_accuracy2 = classifierModel2.validate(classifierModel2.data.valid_dl)
print("Training Accuracy from model 2 is {0}".format(train_accuracy2*100))
print('')
print("Validation Accuracy from model 2 is {0}".format(valid_accuracy2*100))
classifierModel2.show_results(rows=10)
preds2 = classifierModel2.get_preds(DatasetType.Test)
classifierModel2.save('classifierModel2')
build_submission_file(classifierModel2,preds2,'submission2.csv')
classifierModel2.unfreeze()
classifierModel2.fit_one_cycle(1, slice(1e-5/(2.6**4),1e-3))
train_data_lm3 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined1').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
train_data_lm3.save('train_data_lm3.pkl')
train_data_lm3.vocab.itos[:10]
train_data_lm3.train_ds[0][0]
train_data_lm3 = load_data('', 'train_data_lm3.pkl', bs=48)
train_data_lm3.show_batch(5)
languageModel3 = language_model_learner(train_data_lm3, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel3.lr_find()
languageModel3.recorder.plot(skip_end=10)
languageModel3.fit_one_cycle(5, 1e-2)
#languageModel2.save('fine_tuned2')
languageModel3.save_encoder('fine_tuned_enc3')
data_classifier3 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined1', vocab=train_data_lm3.vocab)
                    .split_by_rand_pct(0.1)
                    .label_from_df('variety')
                    .add_test(test_data)
                    .databunch(bs=48))
data_classifier3.save('data_classifier3.pkl')
data_classifier3 = load_data('','data_classifier3.pkl',bs=48)
data_classifier3.show_batch(5)
classifierModel3 = text_classifier_learner(data_classifier3, arch=AWD_LSTM, drop_mult=0.5)
classifierModel3.load_encoder('fine_tuned_enc3')
classifierModel3.freeze()
classifierModel3.summary()
classifierModel3.lr_find()
classifierModel3.recorder.plot()
classifierModel3.fit_one_cycle(1, 2e-2)
for i in range(2,5):
    classifierModel3.freeze_to(-i)
    classifierModel3.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    print ('')
classifierModel3.recorder.plot_losses()
classifierModel3.save('classifierModel3')
train_loss3,train_accuracy3 = classifierModel3.validate(classifierModel3.data.train_dl)
valid_loss3,valid_accuracy3 = classifierModel3.validate(classifierModel3.data.valid_dl)
print("Training Accuracy from model 3 is {0}".format(train_accuracy3*100))
print('')
print("Validation Accuracy from model 3 is {0}".format(valid_accuracy3*100))
classifierModel3.show_results(rows=10)
preds3 = classifierModel3.get_preds(DatasetType.Test)
build_submission_file(classifierModel3,preds3,'submission3.csv')
#classifierModel3.unfreeze()
#classifierModel3.fit_one_cycle(1, slice(1e-5/(2.6**4),1e-3))
train_data_lm4 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined2').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
train_data_lm4.save('train_data_lm4.pkl')
train_data_lm4.vocab.itos[:10]
train_data_lm4.train_ds[0][0]
train_data_lm4 = load_data('', 'train_data_lm4.pkl', bs=48)
train_data_lm4.show_batch(5)
languageModel4 = language_model_learner(train_data_lm4, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel4.lr_find()
languageModel4.recorder.plot(skip_end=10)
languageModel4.fit_one_cycle(5, 1e-2)
#languageModel2.save('fine_tuned2')
languageModel4.save_encoder('fine_tuned_enc4')
data_classifier4 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined2', vocab=train_data_lm4.vocab)
                    .split_by_rand_pct(0.1)
                    .label_from_df('variety')
                    .add_test(test_data)
                    .databunch(bs=48))
data_classifier4.save('data_classifier4.pkl')
data_classifier4 = load_data('','data_classifier4.pkl',bs=48)
data_classifier4.show_batch(5)
classifierModel4 = text_classifier_learner(data_classifier4, arch=AWD_LSTM, drop_mult=0.5)
classifierModel4.load_encoder('fine_tuned_enc4')
classifierModel4.freeze()
classifierModel4.summary()
classifierModel4.lr_find()
classifierModel4.recorder.plot()
classifierModel4.fit_one_cycle(1, 2e-2)
for i in range(2,5):
    classifierModel4.freeze_to(-i)
    classifierModel4.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    print ('')
classifierModel4.recorder.plot_losses()
classifierModel4.save('classifierModel4')
train_loss4,train_accuracy4 = classifierModel4.validate(classifierModel4.data.train_dl)
valid_loss4,valid_accuracy4 = classifierModel4.validate(classifierModel4.data.valid_dl)
print("Training Accuracy from model 4 is {0}".format(train_accuracy4*100))
print('')
print("Validation Accuracy from model 4 is {0}".format(valid_accuracy4*100))
classifierModel4.show_results(rows=10)
preds4 = classifierModel4.get_preds(DatasetType.Test)
build_submission_file(classifierModel4,preds4,'submission4.csv')
#classifierModel4.unfreeze()
#classifierModel4.fit_one_cycle(1, slice(1e-5/(2.6**4),1e-3))
train_data_lm5 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined3').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
train_data_lm5.save('train_data_lm5.pkl')
train_data_lm5.vocab.itos[:10]
train_data_lm5.train_ds[0][0]
train_data_lm5 = load_data('', 'train_data_lm5.pkl', bs=48)
train_data_lm5.show_batch(5)
languageModel5 = language_model_learner(train_data_lm5, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel5.lr_find()
languageModel5.recorder.plot(skip_end=10)
languageModel5.fit_one_cycle(5, 1e-2)
#languageModel2.save('fine_tuned2')
languageModel5.save_encoder('fine_tuned_enc5')
data_classifier5 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined3', vocab=train_data_lm5.vocab)
                    .split_by_rand_pct(0.1)
                    .label_from_df('variety')
                    .add_test(test_data)
                    .databunch(bs=48))
data_classifier5.save('data_classifier5.pkl')
data_classifier5 = load_data('','data_classifier5.pkl',bs=48)
data_classifier5.show_batch(5)
classifierModel5 = text_classifier_learner(data_classifier5, arch=AWD_LSTM, drop_mult=0.5)
classifierModel5.load_encoder('fine_tuned_enc5')
classifierModel5.freeze()
classifierModel5.summary()
classifierModel5.lr_find()
classifierModel5.recorder.plot()
classifierModel5.fit_one_cycle(1, 2e-2)
for i in range(2,5):
    classifierModel5.freeze_to(-i)
    classifierModel5.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    print ('')
classifierModel5.recorder.plot_losses()
classifierModel5.save('classifierModel5')
train_loss5,train_accuracy5 = classifierModel5.validate(classifierModel5.data.train_dl)
valid_loss5,valid_accuracy5 = classifierModel5.validate(classifierModel5.data.valid_dl)
print("Training Accuracy from model 5 is {0}".format(train_accuracy5*100))
print('')
print("Validation Accuracy from model 5 is {0}".format(valid_accuracy5*100))
classifierModel5.show_results(rows=10)
preds5 = classifierModel5.get_preds(DatasetType.Test)
build_submission_file(classifierModel5,preds5,'submission5.csv')
classifierModel5.unfreeze() #more epochs to check if accuracy improves
classifierModel5.fit_one_cycle(1, slice(1e-5/(2.6**4),1e-3))
classifierModel5.unfreeze() #more epochs to check if accuracy improves
classifierModel5.fit_one_cycle(1, slice(1e-5/(2.6**4),1e-3))
train_data_lm6 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined4').split_by_rand_pct(0.1).label_for_lm().databunch(bs=48))
os.remove("/kaggle/working/train_data_lm1.pkl")  #storage space issue on kaggle
os.remove("/kaggle/working/train_data_lm2.pkl")
os.remove("/kaggle/working/train_data_lm3.pkl")
os.remove("/kaggle/working/train_data_lm4.pkl")
train_data_lm6.save('train_data_lm6.pkl')
train_data_lm6.vocab.itos[:10]
train_data_lm6.train_ds[0][0]
train_data_lm6 = load_data('', 'train_data_lm6.pkl', bs=48)
train_data_lm6.show_batch(5)
languageModel6 = language_model_learner(train_data_lm6, arch=AWD_LSTM, pretrained_fnames=fnames, drop_mult=0.3)
languageModel6.lr_find()
languageModel6.recorder.plot(skip_end=10)
languageModel6.fit_one_cycle(5, 1e-2)
#languageModel2.save('fine_tuned2')
languageModel6.save_encoder('fine_tuned_enc6')
data_classifier6 = (TextList.from_df(df=train_data,path='.',cols='all_text_combined4', vocab=train_data_lm6.vocab)
                    .split_by_rand_pct(0.1)
                    .label_from_df('variety')
                    .add_test(test_data)
                    .databunch(bs=48))
data_classifier6.save('data_classifier6.pkl')
data_classifier6 = load_data('','data_classifier6.pkl',bs=48)
data_classifier6.show_batch(5)
classifierModel6 = text_classifier_learner(data_classifier6, arch=AWD_LSTM, drop_mult=0.5,metrics = [accuracy])
classifierModel6.load_encoder('fine_tuned_enc6')
classifierModel6.freeze()
classifierModel6.summary()
classifierModel6.lr_find()
classifierModel6.recorder.plot()
classifierModel6.fit_one_cycle(1, 2e-2)
for i in range(2,5):
    classifierModel6.freeze_to(-i)
    classifierModel6.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    print ('')
classifierModel6.recorder.plot_losses()
classifierModel6.save('classifierModel6')
train_loss6,train_accuracy6 = classifierModel6.validate(classifierModel6.data.train_dl)
valid_loss6,valid_accuracy6 = classifierModel6.validate(classifierModel6.data.valid_dl)
print("Training Accuracy from model 6 is {0}".format(train_accuracy6*100))
print('')
print("Validation Accuracy from model 6 is {0}".format(valid_accuracy6*100))
classifierModel6.show_results(rows=10)
preds6 = classifierModel6.get_preds(DatasetType.Test)
build_submission_file(classifierModel6,preds6,'submission6.csv')
#classifierModel6.unfreeze()
#classifierModel6.fit_one_cycle(1, slice(1e-5/(2.6**4),1e-3))
models =['ClassifierModel1',
         'ClassifierModel2',
         'ClassifierModel3',
         'ClassifierModel4',
         'ClassifierModel5',
         'ClassifierModel6']
cols = ['Training Accuracy',
        'Training Loss',
        'Validation Accuracy',
        'Validation Loss']
                                       
final_results = pd.DataFrame(columns = cols,index=models)
final_results[cols[0]][models[0]] = train_accuracy1*100
final_results[cols[1]][models[0]] = train_loss1
final_results[cols[2]][models[0]] = valid_accuracy1*100
final_results[cols[3]][models[0]] = valid_loss1

final_results[cols[0]][models[1]] = train_accuracy2*100
final_results[cols[1]][models[1]] = train_loss2
final_results[cols[2]][models[1]] = valid_accuracy2*100
final_results[cols[3]][models[1]] = valid_loss2

final_results[cols[0]][models[2]] = train_accuracy3*100
final_results[cols[1]][models[2]] = train_loss3
final_results[cols[2]][models[2]] = valid_accuracy3*100
final_results[cols[3]][models[2]] = valid_loss3

final_results[cols[0]][models[3]] = train_accuracy4*100
final_results[cols[1]][models[3]] = train_loss4
final_results[cols[2]][models[3]] = valid_accuracy4*100
final_results[cols[3]][models[3]] = valid_loss4

final_results[cols[0]][models[4]] = train_accuracy5*100
final_results[cols[1]][models[4]] = train_loss5
final_results[cols[2]][models[4]] = valid_accuracy5*100
final_results[cols[3]][models[4]] = valid_loss5


final_results[cols[0]][models[5]] = train_accuracy6*100
final_results[cols[1]][models[5]] = train_loss6
final_results[cols[2]][models[5]] = valid_accuracy6*100
final_results[cols[3]][models[5]] = valid_loss6

print("Final results")
print('')
print(final_results)
f,ax = plt.subplots(figsize = (20,20))
sns.pointplot(final_results.index,final_results['Validation Accuracy'])
plt.xticks(fontsize =15)
plt.yticks(fontsize =15)
plt.xlabel("Wines",fontsize =20)
plt.ylabel("Models",fontsize =20,rotation = 0)
plt.title('Validation Accuracy', fontsize=25)
plt.grid(True)
plt.show()
