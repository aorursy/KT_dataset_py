import pandas as pd
adult_file_path = '../input/adult.csv'
adult_income_data = pd.read_csv(adult_file_path)
adult_income_data.columns =adult_income_data.columns.str.strip().str.lower().str.replace('.', '_')
adult_income_data.describe()
import seaborn as sns
sns.stripplot(x='sex', y='hours_per_week', data=adult_income_data,hue='income',marker='X')
import seaborn as sns
sns.boxplot(x='hours_per_week',y='marital_status',data=adult_income_data,palette='rainbow',hue='income')
import seaborn as sns
sns.boxplot(x='hours_per_week',y='education',data=adult_income_data,palette='rainbow',hue='income')
import seaborn as sns
sns.boxplot(x='hours_per_week',y='race',data=adult_income_data,palette='rainbow',hue='income')
import seaborn as sns
import matplotlib as plt
sns.stripplot(x='capital_gain', y='sex', data=adult_income_data,hue='income')
#plt.gca().set_xscale('log')
plt.xlim(0, 6000)
sns.stripplot(x='age', y='sex', data=adult_income_data,hue='income')
#plt.gca().set_xscale('log')
#plt.xlim(0, 6000)
import pandas as pd
import numpy as np
money = {'<=50K': 0,">50K": 1} 
ale = {'Female': 0,"Male": 1} 
adult_income_data.income = [money[item] for item in adult_income_data.income] 
adult_income_data.sex = [ale[item] for item in adult_income_data.sex] 
white=[]
black=[]
native_american=[]
single=[]
married=[]
separated=[]
divorced=[]
widowed=[]
highdegree=[]
for i in range(len(adult_income_data.race)):
    white.append(1) if adult_income_data.race[i]=="White" else white.append(0)
    black.append(1) if adult_income_data.race[i]=="Black" else black.append(0)
    native_american.append(1) if adult_income_data.native_country[i]=="United-States" else native_american.append(0)
    single.append(1) if  adult_income_data.marital_status[i]=='Never-married' else single.append(0)
    married.append(1) if  adult_income_data.marital_status[i]=='Married-civ-spouse' else married.append(0)
    separated.append(1) if  adult_income_data.marital_status[i]=='Separated' else separated.append(0)
    divorced.append(1) if  adult_income_data.marital_status[i]=='Divorced' else divorced.append(0)
    widowed.append(1) if  adult_income_data.marital_status[i]=='Widowed' else widowed.append(0)
    highdegree.append(1) if adult_income_data.education[i]=='Masters' else (highdegree.append(1) if adult_income_data.education[i]=='Doctorate' else highdegree.append(0))
adult_income_data['white'] = white
adult_income_data['black'] = black
adult_income_data['born_usa'] = native_american
adult_income_data['single'] =single
adult_income_data['married'] =married
adult_income_data['separated'] =separated
adult_income_data['divorced'] =divorced
adult_income_data['widowed'] =widowed
adult_income_data['highdegree'] =highdegree
adult_features = ['age','sex','education_num','hours_per_week','born_usa','white','black','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
#y=adult_income_data.income
#print(adult_income_data.capital-gain)
data2 = adult_income_data[adult_features]
data2.head()
import matplotlib.pyplot as plt
import seaborn as sns

colormap = plt.cm.magma
plt.figure(figsize=(16,16))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data2.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]

cv = KFold(n_splits=10)            # Number of Cross Validation folds
accuracies = list()
max_features = len(list(X))
depth_range = range(1, max_features)

# Testing max_depths from 1 to max features
for depth in depth_range:
    fold_accuracy = []
    tree_model = DecisionTreeClassifier(max_depth = depth)

    for train_fold, valid_fold in cv.split(X):
        f_train = X.loc[train_fold] # Extract train data with cv indices
        f_valid = X.loc[valid_fold] # Extract valid data with cv indices
        model = tree_model.fit(X = f_train.drop(['income'], axis=1), 
                               y = f_train["income"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['income'], axis=1), 
                                y = f_valid["income"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    #print("Accuracy per fold: ", fold_accuracy, "\n")
    #print("Average accuracy: ", avg)
    #print("\n")
    
# To show the accuracy foe each depth
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

# Create Decision Tree with max_depth = 9
decision_tree = DecisionTreeClassifier(max_depth = 7)
decision_tree.fit(train_X, train_y)

# Predicting results for validation dataset
#score=model.score( val_X, val_y)
#print(score)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     #f = Source(
         f=tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 9,
                              impurity = True,
                              feature_names = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss'],
                              class_names = ['<=50K', '>50K'],
                              rounded = True,
                              filled= True )#)
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= Income', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('tree_income.png')
PImage("tree_income.png")


y_pred=decision_tree.predict(val_X)
print("Accuracy:", decision_tree.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

forest = RandomForestClassifier(random_state=1)
forest.fit(train_X, train_y)
y_pred = forest.predict(val_X)
print("Accuracy:", forest.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

logreg = LogisticRegression()
logreg.fit(train_X, train_y)
y_pred = logreg.predict(val_X)
print("Accuracy:", logreg.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))

from sklearn.svm import SVC 
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

 
svclassifier = SVC()  
svclassifier.fit(train_X, train_y)
y_pred = svclassifier.predict(val_X)
print("Accuracy:", svclassifier.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

adult_features = ['age','sex','education_num','hours_per_week','single','married','separated','divorced','widowed','highdegree','capital_gain','capital_loss','income']
X = adult_income_data[adult_features]
y = X['income']
X2 = X.drop(['income'], axis=1).values 
train_X, val_X, train_y, val_y = train_test_split(X2, y, random_state = 0)

neiclassifier = KNeighborsClassifier(n_neighbors=5)  
neiclassifier.fit(train_X, train_y)
y_pred = neiclassifier.predict(val_X)
print("Accuracy:", neiclassifier.score(val_X, val_y))
print("Precision:", precision_score(val_y, y_pred))
print("Recall:", recall_score(val_y, y_pred))
print("F1:", f1_score(val_y, y_pred))
print("Area under precision Recall:", average_precision_score(val_y, y_pred))