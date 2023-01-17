%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob 
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from skimage.util.montage import montage2d
from skimage.io import imread
base_dir = os.path.join('..', 'input')
mont_paths = glob(os.path.join(base_dir, 'Montgomery', 'MontgomerySet', '*', '*.*'))
shen_paths = glob(os.path.join(base_dir, 'ChinaSet_AllFiles', 'ChinaSet_AllFiles', '*', '*.*'))
print('Montgomery Files', len(mont_paths))
print('Shenzhen Files', len(shen_paths))
all_paths_df = pd.DataFrame(dict(path = mont_paths + shen_paths))
all_paths_df['source'] = all_paths_df['path'].map(lambda x: x.split('/')[2])
all_paths_df['file_id'] = all_paths_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_paths_df['patient_group']  = all_paths_df['file_id'].map(lambda x: x.split('_')[0])

all_paths_df['file_ext'] = all_paths_df['path'].map(lambda x: os.path.splitext(x)[1][1:])
all_paths_df = all_paths_df[all_paths_df.file_ext.isin(['png', 'txt'])]
all_paths_df['pulm_state']  = all_paths_df['file_id'].map(lambda x: int(x.split('_')[-1]))
all_paths_df.sample(5)
clean_patients_df = all_paths_df.pivot_table(index = ['patient_group', 'pulm_state', 'file_id'], 
                                             columns=['file_ext'], 
                                             values = 'path', aggfunc='first').reset_index()
clean_patients_df.sample(5)
from warnings import warn
def report_to_dict(in_path):
    with open(in_path, 'r') as f:
        all_lines = [x.strip() for x in f.read().split('\n')]
    info_dict = {}
    try:
        if "Patient's Sex" in all_lines[0]:
            info_dict['age'] = all_lines[1].split(':')[-1].strip().replace('Y', '')
            info_dict['sex'] = all_lines[0].split(':')[-1].strip()
            info_dict['report'] = ' '.join(all_lines[2:]).strip()
        else:
            info_dict['age'] = all_lines[0].split(' ')[-1].replace('yrs', '').replace('yr', '')
            info_dict['sex'] = all_lines[0].split(' ')[0].strip()
            info_dict['report'] = ' '.join(all_lines[1:]).strip()
        
        info_dict['sex'] = info_dict['sex'].upper().replace('FEMALE', 'F').replace('MALE', 'M').replace('FEMAL', 'F')[0:1]
        if 'month' in info_dict.get('age', ''):
            info_dict.pop('age') # invalid
        if 'day' in info_dict.get('age', ''):
            info_dict.pop('age') # invalid
        elif len(info_dict.get('age',''))>0:
            info_dict['age'] = float(info_dict['age'])
        else:
            info_dict.pop('age')
        return info_dict
    except Exception as e:
        print(all_lines)
        warn(str(e), RuntimeWarning)
        return {}
report_df = pd.DataFrame([dict(**report_to_dict(c_row.pop('txt')), **c_row) 
              for  _, c_row in clean_patients_df.iterrows()])
report_df.sample(5)
sns.pairplot(report_df, hue = 'patient_group')
preview_df = report_df.groupby(['patient_group', 'pulm_state']).apply(lambda x: x.sample(2)).reset_index(drop = True)
fig, m_axs = plt.subplots(2, preview_df.shape[0]//2, figsize = (12, 12))
for c_ax, (_, c_row) in zip(m_axs.flatten(), preview_df.iterrows()):
    c_ax.imshow(imread(c_row['png']), cmap = 'bone')
    c_ax.set_title('{patient_group} Age:{age}\nTuberculosis: {pulm_state}\n{report}'.format(**c_row))
from PIL import Image
montage_df = report_df.groupby(['patient_group', 'pulm_state']).apply(lambda x: x.sample(10)).reset_index(drop = True).sort_values('age')
montage_df['image'] = montage_df['png'].map(lambda x: Image.open(x).resize((512, 512)).convert('RGB'))
def redify(img):
    n_img = np.array(img)
    n_img[:,:,1:3] = 0
    return n_img
out_stack = np.concatenate(montage_df.apply(lambda c_row: 
                                            [np.array(c_row['image']) if c_row['pulm_state']==0 else redify(c_row['image'])]
                                            ,1).values,0)
out_montage = np.stack([montage2d(out_stack[:,:,:,i]).astype(np.uint8) for i in range(out_stack.shape[3])], -1)
fig, ax1 = plt.subplots(1,1, figsize = (20, 20))
ax1.imshow(out_montage)
ax1.axis('off')
fig.savefig('preview.png', dpi = 300)
