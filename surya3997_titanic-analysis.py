# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output, check_call
#print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "./"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


inp_dir = '../input/'

data = pd.read_csv(inp_dir + "train.csv")
print(data)

total_count = data.shape[0]
male_count = data[data.Sex == 'male'].shape[0]
print("Male passengers : ", male_count)
female_count = data[data.Sex == 'female'].shape[0]
print("Female passengers : ", female_count)
survived_num = data[data['Survived'] == 1].shape[0]
print("No of people who survived in titanic : ", survived_num)
died_num = data[data['Survived'] == 0].shape[0]
print("No of people who died in titanic : ", died_num)
survived_male_num = data.query('Survived == 1 & Sex == "male"').shape[0]
print("No of males who survived in titanic : ", survived_male_num)
survived_female_num = data.query('Survived == 1 & Sex == "female"').shape[0]
print("No of females who survived in titanic : ", survived_female_num)
died_male_num = data.query('Survived == 0 & Sex == "male"').shape[0]
print("No of males who died in titanic : ", died_male_num)
died_female_num = data.query('Survived == 0 & Sex == "female"').shape[0]
print("No of females who died in titanic : ", died_female_num)
p1_count = data.query('Pclass == 1').shape[0]
p2_count = data.query('Pclass == 2').shape[0]
p3_count = data.query('Pclass == 3').shape[0]
print(p1_count, p2_count, p3_count)
p1_survived_count = data.query('Pclass == 1 & Survived == 1').shape[0]
p1_dead_count = data.query('Pclass == 1 & Survived == 0').shape[0]
p2_survived_count = data.query('Pclass == 2 & Survived == 1').shape[0]
p2_dead_count = data.query('Pclass == 2 & Survived == 0').shape[0]
p3_survived_count = data.query('Pclass == 3 & Survived == 1').shape[0]
p3_dead_count = data.query('Pclass == 3 & Survived == 0').shape[0]

print(p1_survived_count, p1_dead_count, p2_survived_count, p2_dead_count, p3_survived_count, p3_dead_count)
p1_male_count = data.query('Pclass == 1 & Sex == "male"').shape[0]
p1_female_count = data.query('Pclass == 1 & Sex == "female"').shape[0]
p2_male_count = data.query('Pclass == 2 & Sex == "male"').shape[0]
p2_female_count = data.query('Pclass == 2 & Sex == "female"').shape[0]
p3_male_count = data.query('Pclass == 3 & Sex == "male"').shape[0]
p3_female_count = data.query('Pclass == 3 & Sex == "female"').shape[0]

print(p1_male_count, p1_female_count, p2_male_count, p2_female_count, p3_male_count, p3_female_count)
died_fem_percent = round((died_female_num / died_num) * 100, 2)
print("Out of {} died, {} are female. That is {} percent.".format(died_num, died_female_num, died_fem_percent))
died_male_percent = round((died_male_num / died_num) * 100, 2)
print("Out of {} died, {} are male. That is {} percent.".format(died_num, died_male_num, died_male_percent))
total_died_male_percent = round((died_male_num / male_count) * 100, 2)
print("Out of {} people in titanic, {} males died. That is {} percent.".format(total_count, died_male_num, total_died_male_percent))
total_died_female_percent = round((died_female_num / female_count) * 100, 2)
print("Out of {} people in titanic, {} females died. That is {} percent.".format(total_count, died_female_num, total_died_female_percent))
req_data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
print(req_data)
def encode_target(df, target_column, col_name):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[col_name] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)

df2, targets2 = encode_target(req_data, "Sex", "Sex")
df3, targets3 = encode_target(df2, "Embarked", "Embarked")
df3 = df3.fillna(0)
print(df2)
print(df3)
print(targets2)
print(targets3)
features = list(df2.columns[1:])
print(features)
y = df3["Survived"]
X = df3[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X, y)
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("tree1.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "tree1.dot", "-o", "tree1.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")
        
        
visualize_tree(dt, features)
import io
from scipy import misc

def show_tree(decisionTree, feature_names, file_path):
    dotfile = io.StringIO()
    export_graphviz(decisionTree, out_file=dotfile, feature_names=feature_names)
    pydot.graph_from_dot_data(dotfile.getvalue()).write_png(file_path)
    i = misc.imread(file_path)
    plt.imshow(i)

# To use it
show_tree(dt, features, 'test.png')
with open("tree1.dot", 'w') as f:
     f = export_graphviz(dt,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = features,
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= 1.5" corresponds to "Mr." title', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('sample-out.png')
PImage("sample-out.png")