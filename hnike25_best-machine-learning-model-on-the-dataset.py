import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from matplotlib import cm
sns.set_style('ticks')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))
# Import Dataset
df_dup = pd.read_csv('../input/pdb_data_no_dups.csv')
df_seq = pd.read_csv('../input/pdb_data_seq.csv')
# Merge the two Data set together
df_merge = df_dup.merge(df_seq,how='inner',on='structureId')
df_merge.rename({'macromoleculeType_x':'macromoleculeType',
                                            'residueCount_y':'residueCount'},axis=1,inplace=True)
df_merge.drop(['macromoleculeType_y','residueCount_x'],axis=1,inplace=True)
df_isnull = pd.DataFrame(round((df_merge.isnull().sum().sort_values(ascending=False)/df_merge.shape[0])*100,1)).reset_index()
df_isnull.columns = ['Columns', '% of Missing Data']
df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})
cm = sns.light_palette("skyblue", as_cmap=True)
df_isnull = df_isnull.style.background_gradient(cmap=cm)
df_isnull
print ('The publication of the Reserach on the Strutural Sequence of Protein have continually incresed over the last 4 decades')
df_pub_year = df_merge.dropna(subset=['publicationYear']) #dropping the missing values from the publicationYear only
#graph
x1= df_pub_year.publicationYear.value_counts().sort_index().index
y1 = df_pub_year.publicationYear.value_counts().sort_index().values
trace1 = go.Scatter(
    x=x1,
    y=y1,
    mode = 'lines+markers',
    text = x1,
    marker=dict(
    color='blue'),  
)
layout = go.Layout(
    xaxis=dict(
        title = 'Years',
        range = [1967.9,2018.5],
        autotick=True,  
    ),
    yaxis = dict(
    title = 'Frequency'
    ),
    title = 'Number of Publications Since 1968'
    )

fig = go.Figure(data=[trace1], layout=layout)
py.iplot(fig)

# We will split the ph value into three according to the scientific categorization of the Ph such as Acidic if pH <7
#basic if pH >7 and neutral if pH = 7
def ph_scale (ph):
    if ph < 7 :
        ph = 'Acidic'
    elif ph > 7:
        ph = 'Bacis'
    else:
        ph = 'Neutral'
    return ph
print('The pH Scale are group into 3 Categories: BASIC if [ pH > 7 ], ACIDIC if [ pH < 7 ] and NEUTRAL if pH [ is equal to 7 ]')

#Transform the dataset
df_ph = df_merge.dropna(subset=['phValue']) # dropping missing values in the phValue column only
df_ph['pH_scale'] = df_ph['phValue'].apply(ph_scale)
#Graph
labels= df_ph['pH_scale'].value_counts().index
values = df_ph['pH_scale'].value_counts().values
fig = {
      "data": [
        {
          "values":values,
          "labels":labels,
          "text":'pH Scale',
          "textposition":"inside",
          #"domain": {"x": [0, .33]},
          "textfont": {'size':12,'color':'white'},  
          "name": 'pH Scale',
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        }],
    "layout": {
            "title":"pH Distribution",
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": 'pH Scale',
                    "x": 0.50,
                    "y": 0.5
                }]
            }
        }
py.iplot(fig)                              
# The result of this cell Show the Top 10 most used crystallization method
df_cry_meth = df_merge.dropna(subset=['crystallizationMethod']) # this will drop all missing values in
#the crystallizationMethod column

cry_me = pd.DataFrame(df_cry_meth.crystallizationMethod.value_counts(ascending=False).head(10)).reset_index()
cry_me.columns = ['Crystallization Method','Values']

f,ax = plt.subplots(figsize=(10,8))
cry_me.plot(kind = 'barh',ax=ax,color='gray',legend=None,width= 0.8)
# get_width pulls left or right; get_y pushes up or down
for i in ax.patches:
    ax.text(i.get_width()+.1, i.get_y()+.40, \
            str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  
#Set ylabel
ax.set_yticklabels(cry_me['Crystallization Method'])
# invert for largest on top 
ax.invert_yaxis()
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title ('Top 10 Crystallization Method',color='black',fontsize=16)
sns.despine(bottom=True)
popular_exp_tech = df_merge.experimentalTechnique.value_counts()[:3] # Extract the 3 top used Exp Tech 
popular_exp_tech_df = pd.DataFrame(popular_exp_tech).reset_index()
popular_exp_tech_df.columns=['Experimental Technique','values']
# ADDING A ROW FOR THE ORTHER EXPERIMENTAL TECHNIQUE USED. PLEASE PUT IN MIND THAT TO ORTHER TECHNIQUES 
#IS JUST A GROUP OF THE REST OF THECNIQUES USED
popular_exp_tech_df.loc[3]  = ['OTHER TECHNIQUE', 449]
print ('The X-RAY DIFFRACTION is by far the most used Experimental Technique during the Study of the Protein Sequences')

labels = popular_exp_tech_df['Experimental Technique']
values = popular_exp_tech_df['values']
a = 'Exp Tech'
fig = {
      "data": [
        {
          "values":values,
          "labels":labels,
          "text":a,
          "textposition":"inside",
          #"domain": {"x": [0, .33]},
          "textfont": {'size':12,'color':'white'},  
          "name": a,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          "type": "pie"
        }],
    "layout": {
            "title":"Most Used Experimental Techniques",
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": a,
                    "x": 0.50,
                    "y": 0.5
                }]
            }
        }
py.iplot(fig)                              
print ('There are more than 10 macro molecules used in this dataset but PROTEIN is widely used than the others')

ex = df_merge.macromoleculeType.value_counts()
a = 'Macro Mol Type'
colors = ['SlateGray','Orange','Green','DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue',
        'DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue']
fig = {
      "data": [
        {
          "values":ex.values,
          "labels":ex.index,
          "text":a,
          "textposition":"inside",
          #"domain": {"x": [0, .33]},
          "textfont": {'size':12,'color':'white'},  
          "name": a,
          "hoverinfo":"label+percent+name",
          "hole": .4,
          'marker':{'colors':colors
                   },
          "type": "pie"
        }],
    "layout": {
            "title":"Macro Molecule type Distribution",
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": a,
                    "x": 0.50,
                    "y": 0.5
                }]
            }
        }
py.iplot(fig)                              
#classification distribution
clasific =df_merge.classification.value_counts(ascending=False)
df_class = pd.DataFrame(round(((clasific/df_merge.shape[0])*100),2).head(10)).reset_index()
df_class.columns = ['Classification', 'percent_value']
print('There are {} Unique Classification Types and the top 10 Classification type accounts for more than 50% of the classification in the dataset'.format(df_merge.classification.nunique()))
f,ax = plt.subplots(figsize=(10,8))

df_class.plot(kind = 'barh',ax=ax,color='slategray',legend=None,width= 0.8)
# get_width pulls left or right; get_y pushes up or down
for i in ax.patches:
    ax.text(i.get_width()+.1, i.get_y()+.40, \
            str(round((i.get_width()), 2))+'%', fontsize=12, color='black',alpha=0.8)  
#Set ylabel
ax.set_yticklabels(df_class['Classification'])
# invert for largest on top 
ax.invert_yaxis()
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title ('Top 10 Classification Types',color='black',fontsize=16)
sns.despine(bottom=True)
df_class.Classification.values.tolist()[1:4]
# Reduce the df_merge to df_protein which is compose of macromolecule type [Protein and Protein#RNA]
macrotype = ['Protein','Protein#RNA']
df_protein = df_merge[(df_merge['experimentalTechnique'] =='X-RAY DIFFRACTION') & 
                      (df_merge['macromoleculeType'].isin(macrotype))&
                     (df_merge['classification'].isin(df_class.Classification.values.tolist()[1:4]))]

df_protein.reset_index(drop=True,inplace=True)
columns = ['crystallizationMethod' ,'pdbxDetails', 'publicationYear','phValue','crystallizationTempK']
#Dropping columns with missing value above 15%
df_protein.drop(columns=columns,inplace=True)
# Classification Type that will be used from now on
f,ax= plt.subplots(figsize=(10,5))
sns.countplot('classification',data=df_protein, ax=ax)
ax.set_title('Classification Types Selected',fontsize=14,color='black')
ax.tick_params(length =3,labelsize=11,color='black')
ax.set_xlabel('Classification',color='black',fontsize=13)
sns.despine()
from scipy import stats
from scipy.stats import norm, skew, kurtosis
def stat_kde_plot(input1,input2,input3):
    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))
    sns.kdeplot(df_protein[input1],ax = ax1,color ='blue',shade=True,
                label=("Skewness : %.2f"%(df_protein[input1].skew()),
                       "Kurtosis: %.2f"%(df_protein[input1].kurtosis())))
    sns.kdeplot(df_protein[input2], ax = ax2,color='r',shade=True,
                label=("Skewness : %.2f"%(df_protein[input2].skew()),
                       "Kurtosis: %.2f"%(df_protein[input2].kurtosis())))
    sns.kdeplot(df_protein[input3], ax = ax3,color='gray',shade=True,
                label=("Skewness : %.2f"%(df_protein[input3].skew()),
                       "Kurtosis: %.2f"%(df_protein[input3].kurtosis())))
    axes = [ax1,ax2,ax3]
    input = [input1,input2,input3]
    for j in range(len(axes)):
        axes[j].set_xlabel(input[j],color='black',fontsize=12)
        axes[j].set_title(input[j] + ' Kdeplot',fontsize=14)
        axes[j].axvline(df_protein[input[j]].mean() , color ='g',linestyle = '--')
        axes[j].legend(loc ='upper right',fontsize=12,ncol=2)
    sns.despine()
    return plt.show()

stat_kde_plot('resolution','residueCount','structureMolecularWeight')
for i in ['resolution','residueCount','structureMolecularWeight']:
    df_protein[i] = df_protein[i].map(lambda i: np.log(i) if i > 0 else 0)
stat_kde_plot('resolution','residueCount','structureMolecularWeight')
# Drop all null values from this columns
def stat_plot (input):
    (mu, sigma) = norm.fit(df_protein[input])
    f, (ax1, ax2)= plt.subplots(1,2,figsize=(15,5))
    # Apply the log transformation on the column
    sns.distplot(df_protein[input],ax = ax1,fit=norm,color ='blue',hist=False)
    ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
    ax1.set_ylabel('Frequency')
    ax1.set_title(input +' Distribution',color='black',fontsize=14)
    #Get also the QQ-plot
    res = stats.probplot(df_protein[input], plot=ax2)
    sns.despine()
    return plt.show()
stat_plot('structureMolecularWeight')
stat_plot('residueCount')
stat_plot('resolution')
def box_plot(input):
    g = sns.factorplot(x="classification", y = input,data = df_protein, kind="box",size =4,
                  aspect=2)
    plt.title(input, fontsize=14,color='black')
    return plt.show()

box_plot('residueCount')
box_plot('resolution')
box_plot('structureMolecularWeight')
#class_dict = {'RIBOSOME':1,'HYDROLASE':2,'TRANSFERASE':3} 
class_dict = {'HYDROLASE':1,'TRANSFERASE':2,'OXIDOREDUCTASE':3}
df_protein['class'] = df_protein.classification.map(class_dict)
#Reduce the dataset to only numerical column and clssification column
columns = ['resolution','structureMolecularWeight','densityMatthews','densityPercentSol',
           'residueCount','class']
df_ml = df_protein[columns]
df_ml.dropna(inplace=True)
df_ml.head()
colormap = plt.cm.RdBu
f, ax = plt.subplots(figsize=(18,7))
sns.heatmap(df_ml.corr(),cmap= colormap,annot=True,ax=ax,annot_kws ={'fontsize':12})
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':13}
ax.tick_params(**kwargs)
ax.tick_params(**kwargs,axis='x')
plt.title ('Pearson Correlation Matrix', color = 'black',fontsize=18)
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df_ml.drop('class',axis = 1)
y = df_ml['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardizing the dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
def model_select(classifier):
    cv_result = []
    cv_means = []
    # Cross validate model with Kfold stratified cross val
    kfold = StratifiedKFold(n_splits=5)
    cv_result.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))
    cv_means.append(np.mean(cv_result))
    return cv_means
# Fitting all the models 
model_type = [KNeighborsClassifier(),GaussianNB(),RandomForestClassifier(),
              AdaBoostClassifier(),GradientBoostingClassifier(),DecisionTreeClassifier(),ExtraTreesClassifier()]
model_score = [model_select(i) for i in model_type]
classifier = ['KNeighbors','Naive Bayes','Random Forest', 
             'AdaBoost','Gradient Boosting','Decision Tree','Extra Trees']
# Place result in a data Frame
ml_model = pd.DataFrame(model_score,classifier).reset_index()
ml_model.columns=['Model','acc_score']
ml_model.sort_values('acc_score',ascending = False,inplace=True)
ml_model.reset_index(drop=True,inplace = True)
f, ax = plt.subplots(figsize=(10,8))
sns.barplot('acc_score','Model',data=ml_model, ax=ax,palette='RdBu_r',edgecolor=".2")
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.01, i.get_y()+.55, \
        str(round((i.get_width()), 2)), fontsize=12, color='black') 
kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
ax.tick_params(**kwargs)
x_axis = ax.axes.get_xaxis().set_visible(False)
ax.set_title('Model & Accuracy Score',fontsize=16)
sns.despine(bottom=True)
plt.show()
#Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
# Compute confusion matrix
def single_model(model):
    clf = model
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    conf_mx = confusion_matrix(y_pred,y_test)
    return conf_mx

#plot confusion matrix    
def plot_confusion_matrix(cm, classes,model_name):

    plt.figure(figsize=(10,6))
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix: '+ model_name, fontsize=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",fontsize=12)

    plt.tight_layout()
    plt.ylabel('True label',fontsize=12,color='black')
    plt.xlabel('Predicted label',fontsize=12,color='black' )
    np.set_printoptions(precision=2)
    
    return plt.show()
classes = ['HYDROLASE','TRANSFERASE','OXIDOREDUCTASE']
plot_confusion_matrix(single_model(ExtraTreesClassifier()), classes,'Extra Trees Classifier Model')
plot_confusion_matrix(single_model(RandomForestClassifier()), classes,'Random Forest Classifier Model')
#plot_confusion_matrix(single_model(SVC()), classes,'Support Vector Classifier Model')
def sing_model(model,input):
    clf = model
    clf.fit(X_train,y_train)
    model_fi = clf.feature_importances_
    feat_imp = pd.DataFrame(model_fi,df_ml.columns[:-1]).reset_index()
    feat_imp.columns = ['Features','Importance_score']
    feat_imp.sort_values('Importance_score',ascending=False,inplace=True)
    feat_imp.reset_index(drop=True,inplace = True)
    f, ax = plt.subplots(figsize=(10,8))
    sns.barplot('Importance_score','Features',data=feat_imp, ax=ax,palette='RdBu_r',edgecolor=".2")
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width()+.002, i.get_y()+.45, \
            str(round((i.get_width()), 2)), fontsize=12, color='black') 
    kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}
    ax.tick_params(**kwargs)
    x_axis = ax.axes.get_xaxis().set_visible(False)
    ax.set_title(input +':'+ ' Features Importance Score',fontsize=16)
    sns.despine(bottom=True)
    return plt.show()
sing_model(RandomForestClassifier(),'Random Forest Classifier')
sing_model(ExtraTreesClassifier(),'Extra Trees Clas')
kfold = StratifiedKFold(n_splits=5)
# Generate a simple plot of the test and training learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=4, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure(figsize = (10,5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt.show()

plot_learning_curve(ExtraTreesClassifier(),"Extra Trees Classifier Learning curves",X_train,y_train,cv=kfold)
plot_learning_curve(RandomForestClassifier(),"Random Forest Classifier Learning curves",X_train,y_train,cv=kfold)
plot_learning_curve(GradientBoostingClassifier(),"Gradient Boosting Classifier mearning curves",X_train,y_train,cv=kfold)
