import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#dat = pd.read_csv("mushroom.zip", compression="zip")

dat = pd.read_csv("../input/mushrooms.csv")
pd.set_option("display.max_columns", 25)

dat.head()
#Outcome variable:



%config InlineBackend.figure_format = 'retina'



fig, ax = plt.subplots(figsize=(4,4))

sns.countplot(dat['class'])

ax.set_title("Counts of Mushrooms", fontweight="bold", fontsize=16)

ax.set_xticklabels(labels=['Poisonous', 'Edible'])

ax.set_xlabel("");
import matplotlib.gridspec as gridspec

f, ax = plt.subplots(figsize=(10, 34))

gs = gridspec.GridSpec(11, 2)



#Viz for Cap Shape:

ax = plt.subplot(gs[0,0])

sns.countplot(dat['cap-shape'], hue=dat['class'], ax=ax)

ax.set_title("Counts of Mushrooms by Cap Shape", fontweight="bold", fontsize=14)

ax.set_xlabel("")

ax.set_xticklabels(labels=['convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical'])

ax.legend(['poisonous', 'edible'])



#Viz for Cap Surface:

ax1 = plt.subplot(gs[0, 1])

sns.countplot(dat['cap-surface'], hue=dat['class'], ax=ax1)

ax1.set_title('Counts of Mushrooms by Cap Surface', fontweight="bold", fontsize=14)

ax1.set_xlabel("")

ax1.set_xticklabels(labels=['smooth', 'scaly', 'fibrous', 'grooves'])

ax1.legend(['poisonous', 'edible'], loc="upper right")



#Viz for Cap Color:

ax2 = plt.subplot(gs[1, 0])

sns.countplot(y=dat['cap-color'], hue=dat['class'], ax=ax2)

ax2.set_title("Counts of Mushrooms by Cap Color", fontweight="bold", fontsize=14)

ax2.set_ylabel("")

ax2.set_yticklabels(labels=['brown', 'yellow', 'white', 'gray', 

                            'red', 'pink', 'buff', 'purple', 

                            'cinnamon', 'green'])

ax2.legend(['poisonous', 'edible'], loc='lower right')



#Viz for Bruises:

ax3 = plt.subplot(gs[1,1])

sns.countplot(dat['bruises'], hue=dat['class'], ax=ax3)

ax3.set_title("Counts of Mushrooms by Bruises", fontweight="bold", fontsize=14)

ax3.set_xlabel("")

ax3.set_xticklabels(labels=['True', 'False'])

ax3.legend(['poisonous', 'edible'], loc="upper left")



#Viz for Odor:

ax4 = plt.subplot(gs[2,0])

sns.countplot(y=dat['odor'], hue=dat['class'], ax=ax4)

ax4.set_title("Counts of Mushrooms by Odor", fontweight="bold", fontsize=14)

ax4.set_ylabel("")

ax4.set_yticklabels(['pungent', 'almond', 'anise', 'none', 

                     'foul', 'creosote', 'fishy', 'spicy', 'musty'])

ax4.legend(['poisonous', 'edible'])



#Viz for Gill Attachment:

ax5 = plt.subplot(gs[2,1])

sns.countplot(dat['gill-attachment'], hue=dat['class'], ax=ax5)

ax5.set_title("Counts of Mushrooms by Gill Attachment", fontweight="bold", fontsize=14)

ax5.set_xlabel("")

ax5.set_xticklabels(['free', 'attached'])

ax5.legend(['poisonous', 'edible'])



#Viz for Gill Spacing:

ax6 = plt.subplot(gs[3,0])

sns.countplot(dat['gill-spacing'], hue=dat['class'], ax=ax6)

ax6.set_title("Counts of Mushrooms by Gill Spacing", fontweight="bold", fontsize=14)

ax6.set_xlabel("")

ax6.set_xticklabels(['close', 'crowded'])

ax6.legend(['poisonous', 'edible'])



#Viz for Gill Size:

ax7 = plt.subplot(gs[3,1])

sns.countplot(dat['gill-size'], hue=dat['class'], ax=ax7)

ax7.set_title("Counts of Mushrooms by Gill Size", fontweight="bold", fontsize=14)

ax7.set_xlabel("")

ax7.set_xticklabels(['narrow', 'broad'])

ax7.legend(['poisonous', 'edible'])



#Viz for Gill Color:

ax8 = plt.subplot(gs[4,0])

sns.countplot(y=dat['gill-color'], hue=dat['class'], ax=ax8)

ax8.set_title('Counts of Mushrooms by Gill Color', fontweight="bold", fontsize=14)

ax8.set_ylabel('')

ax8.set_yticklabels(['black', 'brown', 'gray', 'pink', 'white',

                     'chocolate', 'purple', 'red', 'buff',

                     'green', 'yellow', 'orange'])

ax8.legend(['poisonous', 'edible'])



#Viz for Stalk Shape:

ax9 = plt.subplot(gs[4,1])

sns.countplot(dat['stalk-shape'], hue=dat['class'], ax=ax9)

ax9.set_title('Counts of Mushrooms by Stalk Shape', fontweight="bold", fontsize=14)

ax9.set_xlabel("")

ax9.set_xticklabels(['enlarging', 'tapering'])

ax9.legend(['poisonous', 'edible'], loc='upper left')



#Viz for Stalk Root:

ax10 = plt.subplot(gs[5,0])

sns.countplot(dat['stalk-root'], hue=dat['class'], ax=ax10)

ax10.set_title("Counts of Mushrooms by Stalk Root", fontweight="bold", fontsize=14)

ax10.set_xlabel("")

ax10.set_xticklabels(['equal', 'club', 'bulbous', 'rooted', 'unknown'])

ax10.legend(['poisonous', 'edible'])



#Viz for Stalk Surface above Ring:

ax11 = plt.subplot(gs[5,1])

sns.countplot(dat['stalk-surface-above-ring'], hue=dat['class'], ax=ax11)

ax11.set_title("Counts of Mushrooms by Above Ring Surface", fontweight='bold', fontsize=13)

ax11.set_xlabel("")

ax11.set_xticklabels(['smooth', 'fibrous', 'silky', 'scaly'])

ax11.legend(['poisonous', 'edible'])



#Viz for Stalk Surface below Ring:

ax12 = plt.subplot(gs[6,0])

sns.countplot(dat['stalk-surface-below-ring'], hue=dat['class'], ax=ax12)

ax12.set_title('Counts of Mushrooms by Below Ring Surface', fontweight='bold', fontsize=13)

ax12.set_xlabel("")

ax12.set_xticklabels(['smooth', 'fibrous', 'scaly', 'silky'])

ax12.legend(['poisonous', 'edible'])



#Viz for Stalk Color above ring:

ax13 = plt.subplot(gs[6,1])

sns.countplot(y=dat['stalk-color-above-ring'], hue=dat['class'], ax=ax13)

ax13.set_title('Counts of Mushrooms by Above Ring Color', fontweight='bold', fontsize=13)

ax13.set_ylabel('')

ax13.set_yticklabels(['white', 'gray', 'pink', 'brown', 'buff', 'red',

                      'orange', 'cinnamon', 'yellow'])

ax13.legend(['poisonous', 'edible'], loc='lower right')



#Viz for Stalk Color below ring:

ax14 = plt.subplot(gs[7,0])

sns.countplot(y=dat['stalk-color-below-ring'], hue=dat['class'], ax=ax14)

ax14.set_title('Counts of Mushrooms by Below Ring Color', fontweight='bold', fontsize=13)

ax14.set_ylabel('')

ax14.set_yticklabels(['white', 'pink', 'gray', 'buff', 'brown', 'red',

                      'yellow', 'orange', 'cinnamon'])

ax14.legend(['poisonous', 'edible'], loc='lower right')



#Viz for Veil Type:

ax15 = plt.subplot(gs[7,1])

sns.countplot(dat['veil-type'], hue=dat['class'], ax=ax15)

ax15.set_title('Counts of Mushrooms by Veil Type', fontweight='bold', fontsize=14)

ax15.set_xlabel('')

ax15.set_xticklabels(['partial'])

ax15.legend("")



#Viz for Veil Color:

ax16 = plt.subplot(gs[8,0])

sns.countplot(dat['veil-color'], hue=dat['class'], ax=ax16)

ax16.set_title('Counts of Mushrooms by Veil Color', fontweight='bold', fontsize=14)

ax16.set_xlabel('')

ax16.set_xticklabels(['white', 'brown', 'orange', 'yellow'])

ax16.legend(['poisonous', 'edible'])



#Viz for Ring Number:

ax17 = plt.subplot(gs[8,1])

sns.countplot(dat['ring-number'], hue=dat['class'], ax=ax17)

ax17.set_title('Counts of Mushrooms by Ring Number', fontweight='bold', fontsize=14)

ax17.set_xlabel('')

ax17.set_xticklabels(['one', 'two', 'none'])

ax17.legend(['poisonous', 'edible'])



#Viz for Ring Type:

ax18 = plt.subplot(gs[9,0])

sns.countplot(dat['ring-type'], hue=dat['class'], ax=ax18)

ax18.set_title('Counts of Mushrooms by Ring Type', fontweight='bold', fontsize=14)

ax18.set_xlabel('')

ax18.set_xticklabels(['pendant', 'evanescent', 'large', 'flaring', 'none'])

ax18.legend(['poisonous', 'edible'], loc='upper right')



#Viz for Spore Print Color:

ax19 = plt.subplot(gs[9,1])

sns.countplot(y=dat['spore-print-color'], hue=dat['class'], ax=ax19)

ax19.set_title('Counts of Mushrooms by Spore Print Color', fontweight='bold', fontsize=13)

ax19.set_ylabel('')

ax19.set_yticklabels(['black', 'brown', 'purple', 'chocolate', 'white',

                      'green', 'orange', 'yellow', 'buff'])

ax19.legend(['poisonous', 'edible'], loc='lower right')



#Viz for Population:

ax20 = plt.subplot(gs[10,0])

sns.countplot(y=dat['population'], hue=dat['class'], ax=ax20)

ax20.set_title('Counts of Mushrooms by Population', fontweight='bold', fontsize=14)

ax20.set_ylabel('')

ax20.set_yticklabels(['scattered', 'numerous', 'abundant', 'several',

                      'solitary', 'clustered'])

ax20.legend(['poisonous', 'edible'])



#Viz for Habitat:

ax21 = plt.subplot(gs[10,1])

sns.countplot(y=dat['habitat'], hue=dat['class'], ax=ax21)

ax21.set_title('Counts of Mushrooms by Habitat', fontweight='bold', fontsize=14)

ax21.set_ylabel("")

ax21.set_yticklabels(['urban', 'grasses', 'meadows', 'woods', 'paths',

                      'waste', 'leaves'])

ax21.legend(['poisonous', 'edible'], loc='lower right')





plt.tight_layout();
#Creates dictionary to help group cap shapes as described above.

cap_shape_dict = {'x': 'convex', 'f': 'flat', 'k': 'knobbed_conical', 'b': 'bell_sunken',

                  's': 'bell_sunken', 'c': 'knobbed_conical'}



#Creates a series of dummy variables.

cap_shape_dummies = pd.get_dummies(dat['cap-shape'].replace(cap_shape_dict.keys(), 

                                                            cap_shape_dict.values()),

                                                            prefix = 'cap_shape')



#Adds dummy variables to dataset.

dat = pd.concat([dat, cap_shape_dummies], axis=1)



#Drop original cap shape variable.

del dat['cap-shape']



#A similar pattern will be followed in future cells to create and add dummy variables so

#no comments will be made to avoid repetitive statements.
cap_surface_dict = {'y': 'scaly_grooves', 'g': 'scaly_grooves', 's': 'smooth', 'f': 'fibrous'}



cap_surface_dummies = pd.get_dummies(dat['cap-surface'].replace(cap_surface_dict.keys(),

                                                                cap_surface_dict.values()),

                                                                prefix = 'cap_surface')



dat = pd.concat([dat, cap_surface_dummies], axis=1)



del dat['cap-surface']
cap_color_dict = {'n': 'brown', 'g': 'gray', 'e': 'red', 'y': 'yellow', 'w': 'white',

                  'b': 'buff_pink', 'p': 'buff_pink', 'c': 'cinn_purp_green', 

                  'r': 'cinn_purp_green', 'u': 'cinn_purp_green'}



cap_color_dummies = pd.get_dummies(dat['cap-color'].replace(cap_color_dict.keys(),

                                                            cap_color_dict.values()),

                                                            prefix = 'cap_color')



dat = pd.concat([dat, cap_color_dummies], axis=1)



del dat['cap-color']
bruises_dict = {'t': 'True', 'f': 'False'}



bruises_dummies = pd.get_dummies(dat['bruises'].replace(bruises_dict.keys(), 

                                                        bruises_dict.values()),

                                                        prefix='bruises')

                                 

dat = pd.concat([dat, bruises_dummies], axis=1)



del dat['bruises']
odor_dict = {'n': 'none', 'f': 'fo_pu_cr_fi_sp_mu', 's': 'fo_pu_cr_fi_sp_mu', 

             'y': 'fo_pu_cr_fi_sp_mu', 'a': 'almond_anise', 'l': 'almond_anise',

             'p': 'fo_pu_cr_fi_sp_mu', 'c': 'fo_pu_cr_fi_sp_mu', 

             'm': 'fo_pu_cr_fi_sp_mu'}



odor_dummies = pd.get_dummies(dat['odor'].replace(odor_dict.keys(), odor_dict.values()),

                              prefix='odor')



dat = pd.concat([dat, odor_dummies], axis=1)



del dat['odor']
gill_attachment_dict = {'f': 'free', 'a': 'attached'}



gill_attachment_dummies = pd.get_dummies(dat['gill-attachment'].replace(gill_attachment_dict.keys(),

                                                                    gill_attachment_dict.values()),

                                                                    prefix="gill_attachment")



dat = pd.concat([dat, gill_attachment_dummies], axis=1)



del dat['gill-attachment']
gill_spacing_dict = {'c': 'close', 'w': 'crowded'}



gill_spacing_dummies = pd.get_dummies(dat['gill-spacing'].replace(gill_spacing_dict.keys(),

                                                                  gill_spacing_dict.values()),

                                                                  prefix="gill_spacing")



dat = pd.concat([dat, gill_spacing_dummies], axis=1)



del dat['gill-spacing']
gill_size_dict = {'b': 'broad', 'n': 'narrow'}



gill_size_dummies = pd.get_dummies(dat['gill-size'].replace(gill_size_dict.keys(),

                                                            gill_size_dict.values()),

                                                            prefix='gill_size')



dat = pd.concat([dat, gill_size_dummies], axis=1)



del dat['gill-size']
gill_color_dict = {'b': 'buff_green', 'p': 'pink', 'w': 'white', 'n': 'brown', 

                   'g': 'gray', 'h': 'chocolate', 'u': 'purple', 'k': 'black',

                   'e': 'red_orange', 'y': 'yellow', 'o': 'red_orange', 

                   'r': 'buff_green'}



gill_color_dummies = pd.get_dummies(dat['gill-color'].replace(gill_color_dict.keys(),

                                                              gill_color_dict.values()),

                                                              prefix= 'gill_color')



dat = pd.concat([dat, gill_color_dummies], axis=1)



del dat['gill-color']
stalk_shape_dict = {'t': 'tapering', 'e': 'enlarging'}



stalk_shape_dummies = pd.get_dummies(dat['stalk-shape'].replace(stalk_shape_dict.keys(),

                                                                stalk_shape_dict.values()),

                                                                prefix= 'stalk_shape')



dat = pd.concat([dat, stalk_shape_dummies], axis=1)



del dat['stalk-shape']
stalk_root_dict = {'b': 'bulbous', '?': 'unknown', 'e': 'equal_club', 'c': 'equal_club',

                   'r': 'rooted'}



stalk_root_dummies = pd.get_dummies(dat['stalk-root'].replace(stalk_root_dict.keys(),

                                                              stalk_root_dict.values()),

                                                              prefix= 'stalk_root')



dat = pd.concat([dat, stalk_root_dummies], axis=1)



del dat['stalk-root']
above_ring_dict = {'s': 'sm_fi_sc', 'k': 'sm_fi_sc', 'f': 'sm_fi_sc', 'y': 'silky'}



above_ring_dummies = pd.get_dummies(dat['stalk-surface-above-ring'].replace(above_ring_dict.keys(),

                                                                            above_ring_dict.values()),

                                                                            prefix='above_ring_surface')



dat = pd.concat([dat, above_ring_dummies], axis=1)



del dat['stalk-surface-above-ring']
below_ring_dict = {'s': 'sm_fi_sc', 'k': 'sm_fi_sc', 'f': 'sm_fi_sc', 'y': 'silky'}



below_ring_dummies = pd.get_dummies(dat['stalk-surface-below-ring'].replace(below_ring_dict.keys(),

                                                                            below_ring_dict.values()),

                                                                            prefix='below_ring_surface')



dat = pd.concat([dat, below_ring_dummies], axis=1)



del dat['stalk-surface-below-ring']
stalk_color_above_dict = {'w': 'white', 'g': 'gray_red_or', 'p': 'pink', 'n': 'brown',

                          'b': 'buff_cinn_yel', 'e': 'gray_red_or', 'o': 'gray_red_or',

                          'c': 'buff_cinn_yel', 'y': 'buff_cinn_yel'}

                          

stalk_color_above_dummies = pd.get_dummies(dat['stalk-color-above-ring'].replace(

                                                            stalk_color_above_dict.keys(),

                                                            stalk_color_above_dict.values()),

                                                            prefix= 'stalk_color_above_ring')



dat = pd.concat([dat, stalk_color_above_dummies], axis=1)

                          

del dat['stalk-color-above-ring']
stalk_color_below_dict = {'w': 'white', 'g': 'gray_red_or', 'p': 'pink', 'n': 'brown',

                          'b': 'buff_cinn_yel', 'e': 'gray_red_or', 'o': 'gray_red_or',

                          'c': 'buff_cinn_yel', 'y': 'buff_cinn_yel'}



stalk_color_below_dummies = pd.get_dummies(dat['stalk-color-below-ring'].replace(

                                                            stalk_color_below_dict.keys(),

                                                            stalk_color_below_dict.values()),

                                                            prefix= 'stalk_color_below_ring')



dat = pd.concat([dat, stalk_color_below_dummies], axis=1)



del dat['stalk-color-below-ring']
del dat['veil-type']
veil_color_dict = {'w': 'white', 'n': 'non-white', 'o': 'non-white', 'y': 'non-white'}



veil_color_dummies = pd.get_dummies(dat['veil-color'].replace(veil_color_dict.keys(),

                                                              veil_color_dict.values()),

                                                              prefix = 'veil_color')



dat = pd.concat([dat, veil_color_dummies], axis=1)



del dat['veil-color']
ring_num_dict = {'o': 'one', 't': 'none_two', 'n': 'none_two'}



ring_num_dummies = pd.get_dummies(dat['ring-number'].replace(ring_num_dict.keys(),

                                                             ring_num_dict.values()),

                                                             prefix = 'ring_num')



dat = pd.concat([dat, ring_num_dummies], axis=1)



del dat['ring-number']
ring_type_dict = {'p': 'pendant_flaring', 'f': 'pendant_flaring', 'l': 'larger_none',

                  'n': 'larger_none', 'e': 'evanescent'}



ring_type_dummies = pd.get_dummies(dat['ring-type'].replace(ring_type_dict.keys(),

                                                            ring_type_dict.values()),

                                                            prefix = 'ring_type')



dat = pd.concat([dat, ring_type_dummies], axis=1)



del dat['ring-type']
spore_color_dict = {'k': 'black', 'n': 'brown', 'u': 'pur_ora_yel_buff',

                    'o': 'pur_ora_yel_buff', 'y': 'pur_ora_yel_buff',

                    'b': 'pur_ora_yel_buff', 'h': 'chocolate', 'w':'white_green',

                    'r': 'white_green'}



spore_color_dummies = pd.get_dummies(dat['spore-print-color'].replace(spore_color_dict.keys(),

                                                                      spore_color_dict.values()),

                                                                      prefix= 'spore_color')



dat = pd.concat([dat, spore_color_dummies], axis=1)



del dat['spore-print-color']
pop_dict = {'s': 'scattered', 'n': 'numerous_abundant', 'a': 'numerous_abundant',

            'v': 'several', 'y': 'solitary', 'c': 'clustered'}



pop_dummies = pd.get_dummies(dat['population'].replace(pop_dict.keys(),

                                                            pop_dict.values()),

                                                            prefix = 'population')



dat = pd.concat([dat, pop_dummies], axis=1)



del dat['population']
habitat_dict = {'u': 'urban', 'g': 'grasses', 'm': 'meadows', 'd': 'woods',

                'p': 'paths', 'w': 'waste', 'l': 'leaves'}



habitat_dummies = pd.get_dummies(dat['habitat'].replace(habitat_dict.keys(),

                                                        habitat_dict.values()),

                                                        prefix = 'habitat')



dat = pd.concat([dat, habitat_dummies], axis=1)



del dat['habitat']
class_dict = {'p': 'poisonous', 'e': 'edible'}



class_dummies = pd.get_dummies(dat['class'].replace(class_dict.keys(),

                                                    class_dict.values()),

                                                    drop_first=True, 

                                                    prefix="class")



dat = pd.concat([dat, class_dummies], axis=1)



del dat['class']
#First we split the data into training and test sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dat.iloc[:, 0:79], dat.iloc[:, -1])
#Now we will tune the training model parameters for SVC using cross validation:

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC
Cs = [0.001, 0.01, 0.1, 1, 10]

gammas = [0.001, 0.01, 0.1, 1]



param_grid = {'C': Cs, 'gamma': gammas}



svc = SVC()

RSCV = RandomizedSearchCV(svc, param_grid, cv=10, n_jobs=-1) 
RSCV.fit(X_train, y_train)
print(RSCV.best_params_)

print(RSCV.best_score_)
mod = SVC(gamma=0.01)

mod.fit(X_train, y_train)
preds = mod.predict(X_test)
accuracy_score(y_test, preds)
confusion_matrix(y_test, preds)