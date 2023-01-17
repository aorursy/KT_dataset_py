import os

from collections import defaultdict

import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import figure_factory as FF



import scipy.ndimage

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



import random

import pydicom







# matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
INPUT_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression'



trainset = pd.read_csv(f'{INPUT_DIR}/train.csv')

testset = pd.read_csv(f'{INPUT_DIR}/test.csv')

sample_sub = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
print('train_set contained {} rows with unique {} patients'.format(len(trainset), trainset['Patient'].nunique()))

display(trainset.head())



print('test_set contained {} rows with unique {} patients'.format(len(testset), trainset['Patient'].nunique()))

display(testset.head())



print('sample_submission contained {} rows'.format(len(sample_sub)))

display(sample_sub.head())
summ = pd.DataFrame({

    'data': ['train.csv', 'test.csv', 'sample_submission.csv'],

    'rows': [len(trainset), len(testset), len(sample_sub)],

    'patient': [trainset['Patient'].nunique(), testset['Patient'].nunique(), sample_sub['Patient_Week'].nunique()]

})

summ.set_index('data', inplace=True)

display(summ)
def individual_patient(patient_id):

    patient_df = trainset[trainset['Patient'] == patient_id]

#     display(patient_df)

    

    fig = make_subplots(rows=2, cols=1, specs=[[{'type':'table'}], 

                                               [{'secondary_y': True}]])

    

    fig.add_trace(go.Table(header=dict(values=list(patient_df.columns[1:]),

                                       align='center',

                                       fill_color='#3c446a',

                                       font=dict(color='white')),

                           cells=dict(values=[patient_df[i] for i in patient_df.columns[1:]],

                                      align='center'),

                           columnwidth = [50, 50, 100, 50, 50, 100]), row=1, col=1)

    

    fig.add_trace(go.Scatter(x=patient_df['Weeks'], 

                             y=patient_df['FVC'], 

                             mode='lines+markers+text', 

                             text=patient_df['FVC'], 

                             name='FVC'), row=2, col=1, secondary_y=False)

    

    fig.add_trace(go.Scatter(x=patient_df['Weeks'],

                             y=patient_df['Percent'],

                             mode='markers', 

                             text=round(patient_df['Percent'], 2),

                             name='Percent'), row=2, col=1, secondary_y=True)

    

#     fig.update_traces(textposition='top center')

    fig.update_layout(title_text=f'<b>FVC</b> (line) and <b>Percent</b> (marker) of patient : <b>{patient_id}</b>',

                      xaxis_title="Weeks",

                      width=800,

                      height=700)

    fig.update_yaxes(title_text="Forced vital capacity", secondary_y=False)

    fig.update_yaxes(title_text="Percent", secondary_y=True)

    

    fig.show()

    
for ss in trainset['SmokingStatus'].unique():

    for sample in random.sample(trainset[trainset['SmokingStatus'] == ss]['Patient'].tolist(), 2):

        individual_patient(sample)
fig = px.histogram(trainset, x='Age', color='Sex', marginal='box', 

                   histnorm='probability density', opacity=0.7)

fig.update_layout(title='Distribution of Age between Male and Female',

                  width=800, height=500)

fig.show()
parti_patient = trainset.drop_duplicates(subset='Patient')





fig = px.histogram(parti_patient,

                  x='Age',

                  facet_row='SmokingStatus',

                  facet_col='Sex',

                  )

fig.for_each_annotation(lambda a: a.update(text=a.text.replace("SmokingStatus=", "")))

fig.update_layout(title='Distribution of Age sperated by Sex (col) and Smoking Status (row)',

                  autosize=True, width=800, height=600,

                  font_size=14)

fig.show()
# age of each smoking status categorized by sex

m_exsmk_age = trainset.query('Sex == "Male" and SmokingStatus == "Ex-smoker"').drop_duplicates(subset='Patient')['Age']

m_cursmk_age = trainset.query('Sex == "Male" and SmokingStatus == "Currently smokes"').drop_duplicates(subset='Patient')['Age']

m_nevsmk_age = trainset.query('Sex == "Male" and SmokingStatus == "Never smoked"').drop_duplicates(subset='Patient')['Age']





f_exsmk_age = trainset.query('Sex == "Female" and SmokingStatus == "Ex-smoker"').drop_duplicates(subset='Patient')['Age']

f_cursmk_age = trainset.query('Sex == "Female" and SmokingStatus == "Currently smokes"').drop_duplicates(subset='Patient')['Age']

f_nevsmk_age = trainset.query('Sex == "Female" and SmokingStatus == "Never smoked"').drop_duplicates(subset='Patient')['Age']





# for pie chart

pie_labels = ['Male & Ex-smoker', 'Male & Currently smokes', 'Male & Never smoked','Female & Ex-smoker', 'Female & Currently smokes', 'Female & Never smoked']

ss_values = [m_exsmk_age, m_cursmk_age, m_nevsmk_age,

             f_exsmk_age, f_cursmk_age, f_nevsmk_age]

pie_values = [*map(lambda x : len(x), ss_values)]
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], subplot_titles=['Male', 'Female'])

fig.add_trace(go.Pie(labels=pie_labels[:3], values=pie_values[:3], ), row=1, col=1)

fig.add_trace(go.Pie(labels=pie_labels[3:], values=pie_values[3:]), row=1, col=2)

fig.update_layout(title='Rate of Ex-smoker, Never smoked, Currently smokes between Male & Female',

                  autosize=True, width=800, height=400)

fig.update_traces(hoverinfo='label', textinfo='percent+value')

fig.show()
pivot_smkstat_sex = pd.pivot_table(trainset, index=['Sex', 'SmokingStatus'], aggfunc={'Age': ['max', 'min', np.mean, np.std],

                                                                                      'FVC': ['max', 'min', np.mean, np.std],

                                                                                      'Percent':['max', 'min', np.mean, np.std]})

display(pivot_smkstat_sex)
age_range = pd.cut(trainset['Age'], np.arange(40, 100, 10))

pivot_smkstat_sex = pd.pivot_table(trainset, index=['Sex', 'SmokingStatus', age_range], aggfunc={'FVC': ['max', 'min', np.mean, np.std],

                                                                                                 'Percent':['max', 'min', np.mean, np.std]})

mean_of_fvc = pivot_smkstat_sex['FVC'][['mean','std']].round(2).reset_index()

mean_of_fvc = mean_of_fvc.sort_values(['Sex','SmokingStatus', 'Age'])

mean_of_fvc['Age'] = mean_of_fvc['Age'].astype(str).map({'(40, 50]':'40_50', '(50, 60]':'50_60', '(60, 70]':'60_70',

                                                         '(70, 80]':'70_80','(80, 90]':'80_90'})

mean_of_fvc = mean_of_fvc.rename(columns={"mean":"mean_FVC", 'std':'std_FVC', 'Age':'AgeRange'})

display(mean_of_fvc)
fig = px.bar(mean_of_fvc,

             x='SmokingStatus',

             y='mean_FVC', color='Sex',

             barmode='group',

             facet_col='AgeRange',

             error_y='std_FVC',

             category_orders={'Age':['40_50', '50_60', '60_70', '70_80', '80_90']}

            )

fig.update_layout(title='Mean of FVC catagorized by Sex and Age range',

                  yaxis_title="Mean value of Forced vital capacity ",

                  width=800,

                  height=500)

fig.show()
age_range_df = trainset.melt(id_vars=['Sex','Age','Weeks','SmokingStatus'], value_vars=['FVC'])

age_range_df['AgeRange'] = pd.cut(age_range_df['Age'], np.arange(40, 100, 10))

age_range_df = age_range_df.loc[:,['Sex','Age','Weeks','SmokingStatus','AgeRange','value']]

age_range_df['AgeRange'] = age_range_df['AgeRange'].astype(str).map({'(40, 50]':'40_50', '(50, 60]':'50_60', '(60, 70]':'60_70',

                                                                     '(70, 80]':'70_80','(80, 90]':'80_90'})
fig = px.scatter(age_range_df,

                 x='Weeks',

                 y='value',

                 facet_col='AgeRange',

                 facet_row='SmokingStatus',

                 category_orders={'AgeRange':['40_50', '50_60', '60_70', '70_80', '80_90']},

                 opacity=0.5,

                 )

fig.for_each_annotation(lambda a: a.update(text=a.text.replace("SmokingStatus=", "")))

fig.update_layout(title='Scatter plot of Age range and Smoking Status')

fig.show()
fig = px.density_contour(trainset,

                         x ='Percent',

                         y ='FVC',

                         marginal_x="histogram",

                         marginal_y="histogram",

                         color='SmokingStatus',

                         

)

fig.update_layout(title='Relationship between Percent and FVC',

                  width=800,

                  height=400)

fig.show()
below_100 = trainset.query('Percent < 100')

more_100 = trainset.query('Percent > 100')

between_5 = trainset.query('97.5 <= Percent <= 102.5')



x_bar = below_100.groupby('SmokingStatus').size().index.to_list()
fig = make_subplots(rows=1,

                    cols=3,

                    subplot_titles=['Percent < 100%', '97.5% <= Percent <= 102.5%', 'Percent > 100%'])

fig.add_trace(go.Bar(x=x_bar,

                     y=below_100.groupby('SmokingStatus').size(),

                     ), row=1, col=1)

fig.add_trace(go.Bar(x=x_bar,

                     y=between_5.groupby('SmokingStatus').size(),

                     ), row=1, col=2)

fig.add_trace(go.Bar(x=x_bar,

                     y=more_100.groupby('SmokingStatus').size(),

                     ), row=1, col=3)





fig.update_layout(title='Count plot of percent region',

                  width=800,

                  height=500,

                  showlegend=False)
trainset.head()
trainset.groupby('')
# df = px.data.gapminder()

# px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",

#            size="pop", color="continent", hover_name="country",

#            log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])



DICOM_DIR = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'



dicom_dict = defaultdict(list)





for dirname in os.listdir(DICOM_DIR):

    path = os.path.join(DICOM_DIR, dirname)

    dicom_dict[dirname].append(path)

    

p_id = sorted(trainset['Patient'].unique())
# ### load_scan:

# 1. take a string path where patient dicom files were stored.

# 2. store every slice into list and sort in ImagePositionPatient order.



# ### dicom_file:

# 1. take index number of patient which stored in dict_dicom earlier

# 2. this might be useful when you need to pick some random patient

# 3. It also takes specific patient Id in case you need.

# 4. Note that this function is going to read all file in taken path.



# ### get_pixels_hu

# 1. take dicom file which had called through dicom_file function

# 2. It stacks up all the load slices of certain patient

# 3. stacked slices will be calculated into Hounsfield Units







def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

        

    return slices



def dicom_file(idx_num, patient_id=None):

    if patient_id:

        return load_scan(dicom_dict[patient_id][0])

    return load_scan(dicom_dict[p_id[idx_num]][0])



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slbice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
test = dicom_file(0)

test_hu = get_pixels_hu(test)

print('Patient {}'.format(test[0].PatientName))

print('Slices : {}\nPixels : ({} x {})'.format(test_hu.shape[0], test_hu.shape[1], test_hu.shape[2]))

# interactive plot is too heavy

# fig = go.Figure()

# fig.add_trace(go.Histogram(x=test_hu.flatten(), nbinsx=80, histnorm='percent'))

# fig.update_layout(title='Hounsfield Units(HU) of Patient ID00007637202177411956430',

#                   width=800,

#                   height=600)



plt.figure(figsize=(12, 8))

ax = sns.distplot(test_hu.flatten(), bins=80, norm_hist=True)

ax.set_title('Hounsfield Units of patient ID00007637202177411956430', fontsize=25)

plt.show()
def show_dicom_pic(dicom_pixel, p_id=None):

    fig = plt.figure(figsize=(18, len(dicom_pixel)//2))

    for idx, pic in enumerate(dicom_pixel):

        fig.add_subplot(len(dicom_pixel)//5, 5, idx+1)

        plt.imshow(pic, cmap='gray')

        plt.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.3)

        plt.title(str(idx + 1))

        plt.colorbar()

    if p_id:

        fig.suptitle('Patient {}'.format(p_id), fontsize=25)

    fig.show()
show_dicom_pic(test_hu)
test2 = dicom_file(40)

test2_hu = get_pixels_hu(test2)

print('Patient {}'.format(test2[0].PatientName))

print('Slices : {}\nPixels : ({} x {})'.format(test2_hu.shape[0], test2_hu.shape[1], test2_hu.shape[2]))



plt.figure(figsize=(12, 8))

ax = sns.distplot(test2_hu.flatten(), bins=80, norm_hist=True)

ax.set_title('Hounsfield Units of patient {}'.format(test2[0].PatientName), fontsize=25)

plt.show()
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)



    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    

    return image, new_spacing
def make_mesh(image, threshold):

    p = image.transpose(2, 1, 0)

    

    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

    return verts, faces



def static_3d(image, threshold=-300):

    

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    

    verts, faces = make_mesh(image, threshold)

    x, y, z = zip(*verts)

    

    mesh = Poly3DCollection(verts[faces], alpha=0.1)

    face_color = [0.5, 0.5, 1]

    mesh.set_facecolor(face_color)

    

    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))

    ax.set_ylim(0, max(y))

    ax.set_zlim(0, max(z))

    plt.show()

    

def interactive_3d(image, threshold=-300):

    verts, faces = make_mesh(image, threshold)

    x, y, z = zip(*verts)

    fig = FF.create_trisurf(x=x,

                            y=y,

                            z=z,

                            plot_edges=False,

                            simplices=faces)

    iplot(fig)
resampled_test2_hu, spacing = resample(test2_hu, test2)
static_3d(resampled_test2_hu)
def largest_label_volume(im, bg=-1):

    vals, counts = np.unique(im, return_counts=True)



    counts = counts[vals != bg]

    vals = vals[vals != bg]



    if len(counts) > 0:

        return vals[np.argmax(counts)]

    else:

        return None



def segment_lung_mask(image, fill_lung_structures=True):

    

    # not actually binary, but 1 and 2. 

    # 0 is treated as background, which we do not want

    binary_image = np.array(image > -320, dtype=np.int8)+1

    labels = measure.label(binary_image)

    

    # Pick the pixel in the very corner to determine which label is air.

    #   Improvement: Pick multiple background labels from around the patient

    #   More resistant to "trays" on which the patient lays cutting the air 

    #   around the person in half

    background_label = labels[0,0,0]

    

    #Fill the air around the person

    binary_image[background_label == labels] = 2

    

    

    # Method of filling the lung structures (that is superior to something like 

    # morphological closing)

    if fill_lung_structures:

        # For every slice we determine the largest solid structure

        for i, axial_slice in enumerate(binary_image):

            axial_slice = axial_slice - 1

            labeling = measure.label(axial_slice)

            l_max = largest_label_volume(labeling, bg=0)

            

            if l_max is not None: #This slice contains some lung

                binary_image[i][labeling != l_max] = 1

    return binary_image
segmented_lungs = segment_lung_mask(resampled_test2_hu, False)

segmented_lungs_fill = segment_lung_mask(resampled_test2_hu, True)
static_3d(segmented_lungs, 1.5)
static_3d(segmented_lungs_fill, 1.5)
static_3d(segmented_lungs_fill - segmented_lungs, -0.5)
interactive_3d(segmented_lungs_fill - segmented_lungs, -0.5)