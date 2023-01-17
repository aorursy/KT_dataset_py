import numpy as np 

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt

import pydicom



import warnings

warnings.filterwarnings("ignore")
PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

raw_train = pd.read_csv(PATH + 'stage_1_train.csv')

raw_train.head(12)
null_num = pd.isnull(raw_train.Label).values.any()

na_num = pd.isna(raw_train.Label).values.any()

print('isNull:', null_num)

print('isNA:', null_num)
train = raw_train.copy()

train['Subtype'] = train['ID'].apply(lambda st: st.split('_')[2])

train['ID'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1])

train = train[['ID','Subtype','Label']]

train.head(6)
sns.set("notebook",

        font_scale=2.0, 

        rc={'axes.facecolor':'white', 

            'figure.facecolor':'white',

            "grid.linewidth": 1,

            'grid.color': 'gray'})



plt.figure(figsize = (18,6),dpi=250)

ax = sns.countplot(x="Subtype", hue="Label", data=train, palette="ocean", saturation=1.0)

ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

ax.legend(title='Label', bbox_to_anchor=(0.5, 0.0, 0.65, 0.9))

ax.set_xlabel('Subtype',fontdict={'fontweight':'bold'})

ax.set_ylabel('Count',fontdict={'fontweight':'bold'})

for p in ax.patches:

    ax.annotate(p.get_height(), (p.get_x()+0.05, p.get_height()+20000), fontsize=14)

plt.show() 
#reshape dataframe

train_pivot = train.drop_duplicates().pivot(index='ID', columns = 'Subtype', values = 'Label').reset_index()
# list subtypes name to array list

subtype=train_pivot.select_dtypes(include=int).columns.tolist()

subtype.remove('any')
# count number of subtype positive in each image

train_pivot['subtype_num'] = train_pivot[subtype].sum(axis=1, skipna=True)

# show head of new dataframe

train_pivot.head()
# read number of images

data_num = len(train_pivot)

print('Total number of ct image = ', data_num)
# separate negative to subtype of interest/ positive to other subtypes (all negative)and create new dataframe

all_negative = train_pivot.loc[train_pivot['subtype_num']==0].reset_index(drop=True)

all_negative['group'] = 'This subtype :  - , Others subtypes:  -'

all_negative.tail()
# separate dataframe of each subtype

gbl = globals()

for i in subtype:

    # separate positive to subtype of interest/ positive to other subtypes

    gbl[i+'_positive_others_positive']=train_pivot.loc[(train_pivot[i]==1)&(train_pivot['subtype_num']>1)].reset_index(drop=True)

    gbl[i+'_positive_others_positive']['group'] = 'This subtype : + , Others subtypes: +'

    # separate negative to subtype of interest/ positive to other subtypes

    gbl[i+'_positive_others_negative']=train_pivot.loc[(train_pivot[i]==1)&(train_pivot['subtype_num']==1)].reset_index(drop=True)

    gbl[i+'_positive_others_negative']['group'] = 'This subtype : + , Others subtypes:  -'

    # separate positive to subtype of interest/ negative to other subtypes

    gbl[i+'_negative_others_positive']=train_pivot.loc[(train_pivot[i]==0)&(train_pivot['subtype_num']>=1)].reset_index(drop=True)

    gbl[i+'_negative_others_positive']['group'] = 'This subtype :  - , Others subtypes: +'

    # merge 4 groups into one dataframe

    gbl[i] = pd.concat([gbl[i+'_positive_others_positive'],

                        gbl[i+'_positive_others_negative'], 

                        gbl[i+'_negative_others_positive'],

                        all_negative])

    # rename dataframe

    gbl[i].name = i
epidural
# read information of window from dicom file

def read_value(dicom, tag):

    value = dicom[tag].value

    if type(value) == pydicom.multival.MultiValue:

        return int(value[0])

    else:

        return int(value)

    

def read_window(dicom):

    tags = [('0028','1050'), #window center tag

            ('0028','1051'), #window width tag

            ('0028','1052'), #intercept tag

            ('0028','1053')] #slope tag

    return [read_value(dicom, x) for x in tags]



# use window properties to convert scaled pixel array to Hounsfield unit

def apply_linear_transform(pixel_array, center, width, intercept, slope):

    HU = (pixel_array*slope) + intercept

    HU_max = center + (width//2)

    HU_min = center - (width//2)

    HU[HU>HU_max] = HU_max  

    HU[HU<HU_min] = HU_min

    return HU 
# create report of each subtype (graph showing percentages of +/+, +/-, -/+, -/- and images of +/- )

def report(subtype):

    # calculate percentage of each group

    distribution=subtype.groupby('group').size()

    data = distribution.to_list()

    data = list(map(lambda x:x*100/data_num,data))



    fig = plt.figure(figsize=(6, 3),

                       dpi=250)   



    #create pie graph showing percentage of each group

    ax1 = fig.add_axes([-0.3,0,2,2])

    fig.subplots_adjust(wspace=0)



    wedges = ax1.pie(data, 

            startangle=30,

            colors = ['#191970', '#0071C6', '#00C69C','#00FF80'],

            radius=1.0,

            wedgeprops=dict(width=0.3, edgecolor='w'))



    ax1.legend(wedges, labels=distribution.index,

          loc="right",

          fontsize=16,

          framealpha=0,

          labelspacing=0.13,

          bbox_to_anchor=(0.5, 0.5, 1.5, 0.3)

         )



    textstr = '\n'

    

    for i in data:

        percentage = str(round(i,2))

        textstr = textstr + percentage + '%\n'



    props = dict(boxstyle='round', alpha=0)

    ax1.text(1.0, 0.8, subtype.name, transform=ax1.transAxes, fontdict={ 'weight': 'bold','size': 28 }, bbox=props)

    ax1.text(1.0, 0.535, textstr, transform=ax1.transAxes, fontsize=16,bbox=props)  

    

    #print images of +/-

    file = PATH + 'stage_1_train_images/' + gbl[subtype.name+'_positive_others_negative']['ID']

    rows =1 

    columns = 4

    for i in range(1, (rows*columns)+1):

        dicom = pydicom.read_file(file[i]+'.dcm')

        pixel = dicom.pixel_array

        # read window properties

        center , width, intercept, slope = read_window(dicom)

        # use window properties to convert scaled pixel array to Hounsfield unit

        image = apply_linear_transform(pixel, center, width, intercept, slope)

        x = 1.08 + ((i-1)*0.3)

        y = 0.4

        width = 1/columns;

        ax2 = fig.add_axes([x,y,0.5,0.5])

        ax2.axis('off')

        plt.imshow(image, cmap=plt.cm.inferno)

    text = 'some images of ' + subtype.name + ' +, other subtypes -'

    ax3 = fig.add_axes([1.08,0.4,0.5,0.5])

    ax3.text(0.0, 1.1, text, transform=ax3.transAxes, fontdict={'size': 14})

    

    plt.tight_layout()

    plt.show()
# print report

subtype = [epidural, intraparenchymal, intraventricular, subarachnoid, subdural]

for i in subtype:

    report(i)