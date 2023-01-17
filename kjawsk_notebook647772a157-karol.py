from fastai.tabular import *
import torch.nn.functional as F
from datetime import datetime
import pandas as pd
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df.head()
has_day = lambda x: True if 'Day' in str(x) or 'day' in str(x) else False
df['Age upon Outcome'].apply(has_day).any()
def age_upon_outcome(x):
    if not isinstance(x, str):
        return 0;
    if 'month' in x:
        return 4*int(re.search(r'\d+', x).group());
    elif 'year' in x:
        return 12*4*int(re.search(r'\d+', x).group());
    elif 'week' in x:
        return int(re.search(r'\d+', x).group());
    else:
        return 0;
df_age_upon = df['Age upon Outcome'].apply(lambda age_upon: age_upon_outcome(age_upon))
df_age_upon_test = df_test['Age upon Outcome'].apply(lambda age_upon: age_upon_outcome(age_upon))
df['Age upon Outcome'] = df_age_upon;
df_test['Age upon Outcome'] = df_age_upon_test;
df.Breed.unique()
is_mix = lambda x: 1 if 'Mix' in x else 0
df['Breed_is_mixed'] = df['Breed'].apply(is_mix)
df_test['Breed_is_mixed'] = df_test['Breed'].apply(is_mix)
df.head()
is_domestic = lambda x: 1 if 'Domestic' in x else 0
df['Breed_is_domestic'] = df['Breed'].apply(is_domestic)
df_test['Breed_is_domestic'] = df_test['Breed'].apply(is_domestic)
is_shorthair = lambda x: 1 if 'Shorthair' in x else 0
df['Breed_is_shorthair'] = df['Breed'].apply(is_shorthair)
df_test['Breed_is_shorthair'] = df_test['Breed'].apply(is_shorthair)
is_mediumhair = lambda x: 1 if 'Medium Hair' in x else 0
df['Breed_is_mediumhair'] = df['Breed'].apply(is_mediumhair)
df_test['Breed_is_mediumhair'] = df_test['Breed'].apply(is_mediumhair)
is_longhair = lambda x: 1 if 'Longhair' in x else 0
df['Breed_is_longhair'] = df['Breed'].apply(is_longhair)
df_test['Breed_is_longhair'] = df_test['Breed'].apply(is_longhair)
df.head()
df.Color.unique()
len(df.Color.unique())
is_color_mix = lambda x: 1 if '/' in x or 'Tricolor' in x else 0
df['Color_is_mixed'] = df['Color'].apply(is_color_mix)
df_test['Color_is_mixed'] = df_test['Color'].apply(is_color_mix)
df.head()
color1 = []
color2 = []
color1_test = []
color2_test = []
colors = df['Color'].apply(lambda x: x.split('/'));
colors_test = df_test['Color'].apply(lambda x: x.split('/'));
colors.apply(lambda x: color1.append(x[0]));
colors_test.apply(lambda x: color1_test.append(x[0]));
colors.apply(lambda x: color2.append(x[0]) if len(x) == 2 else color2.append('N/A'));
colors_test.apply(lambda x: color2_test.append(x[0]) if len(x) == 2 else color2_test.append('N/A'));
df['Color1'] = color1
df['Color2'] = color2
df_test['Color1'] = color1_test
df_test['Color2'] = color2_test
add_datepart(df, 'DateTime');
add_datepart(df, 'Date of Birth');
add_datepart(df_test, 'DateTime');
add_datepart(df_test, 'Date of Birth');
has_name = lambda x: 0 if str(x) == "nan" else 1
df['Has_Name'] = df['Name'].apply(has_name)
df_test['Has_Name'] = df_test['Name'].apply(has_name)
df.head()
df['Animal ID'].duplicated().any()
df['Duplicated Animal ID'] = df['Animal ID'].duplicated(keep=False)
df_test['Duplicated Animal ID'] = df['Animal ID'].duplicated(keep=False)
df.duplicated().any()
# dep_var = 'Outcome Type'
# cat_names = ['Has_Name', 'Animal Type', 'Sex upon Outcome', 'Breed', 'Color', 'DateTimeMonth',
#              'DateTimeWeek', 'DateTimeDay', 'DateTimeDayofweek', 'DateTimeDayofyear', 'DateTimeIs_month_end',
#              'DateTimeIs_month_start', 'DateTimeIs_quarter_end', 'DateTimeIs_quarter_start', 'DateTimeIs_year_end',
#              'DateTimeIs_year_start', 'Date of BirthMonth', 'Date of BirthWeek', 'Date of BirthDay', 'Date of BirthDayofweek',
#              'Date of BirthDayofyear', 'Date of BirthIs_month_end', 'Date of BirthIs_month_start', 'Date of BirthIs_quarter_end',
#              'Date of BirthIs_quarter_start', 'Date of BirthIs_quarter_start', 'Date of BirthIs_year_end', 'Date of BirthIs_year_start',
#              'Breed_is_mixed', 'Color_is_mixed', 'Color1', 'Color2', 'Breed_is_mixed', 'Breed_is_domestic', 'Breed_is_shorthair', 'Breed_is_mediumhair', 'Breed_is_longhair',
#              'Duplicated Animal ID'
# ]
# cont_names = ['Age upon Outcome', 'DateTimeYear', 'DateTimeElapsed', 'Date of BirthYear', 'Date of BirthElapsed']
# procs = [FillMissing, Categorify, Normalize]
dep_var = 'Outcome Type'
cat_names = ['Animal ID', 'Has_Name', 'Animal Type', 'Sex upon Outcome', 'Breed', 'Color', 'DateTimeMonth',
             'DateTimeWeek', 'DateTimeDay', 'DateTimeDayofweek', 'DateTimeDayofyear', 'Date of BirthMonth', 'Date of BirthWeek', 'Date of BirthDay',
             'Color1', 'Color2', 'Breed_is_domestic', 'Breed_is_shorthair', 'Breed_is_mediumhair', 'Breed_is_longhair', 'Duplicated Animal ID'
]
cont_names = ['Age upon Outcome', 'DateTimeYear', 'DateTimeElapsed', 'Date of BirthYear', 'Date of BirthElapsed']
procs = [FillMissing, Categorify, Normalize]
test_list = TabularList.from_df(df_test, cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(df, path='train.csv', cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(0,20000)))
                           .label_from_df(cols=dep_var)
                           .add_test(test_list)
                           .databunch())
learn = tabular_learner(data, layers=[200,100], metrics=[accuracy, F.cross_entropy])
learn.fit_one_cycle(1)

submission = pd.DataFrame(columns= ['ID', 'Adoption', 'Died', 'Disposal', 'Euthanasia', 'Missing', 'Relocate', 'Return to Owner', 'Rto-Adopt', 'Transfer']); submission.head()
submission['ID'] = df_test['ID']
preds, _ = learn.get_preds(DatasetType.Test)
adoption = []
died = []
disposal = []
euthanasia = []
missing = []
relocate = []
return_owner = []
adpot = []
transfer = []
for row in preds:
    adoption.append(row[0].item())
    died.append(row[1].item())
    disposal.append(row[2].item())
    euthanasia.append(row[3].item())
    missing.append(row[4].item())
    relocate.append(row[5].item())
    return_owner.append(row[6].item())
    adpot.append(row[7].item())
    transfer.append(row[8].item())
submission['Adoption'] = adoption
submission['Died'] = died
submission['Disposal'] = disposal
submission['Euthanasia'] = euthanasia
submission['Missing'] = missing
submission['Relocate'] = relocate
submission['Return to Owner'] = return_owner
submission['Rto-Adopt'] = adpot
submission['Transfer'] = transfer
submission.head()
submission.to_csv("submission.csv", index=False)
