# Libraries Needed
import pandas as pd
import os
from glob import glob
from collections import namedtuple
from pathlib import Path
from pandas_profiling import ProfileReport


# Some display options to easily eyeball the dataframes and their contents
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 1000)
Dataset = namedtuple('Dataset', ['df', 'source'])
datasets = {}

# Reading all csv files as Datasets
for dirname, _, filenames in os.walk('/kaggle/input'):
    for csv_path in glob(os.path.join(dirname, "*.csv")):
        file_name = os.path.basename(csv_path)
        name = os.path.splitext(file_name)[0]
        source = os.path.basename(Path(csv_path).parent)
        datasets[name] = Dataset(pd.read_csv(csv_path, low_memory=False), source)

print("Read a total of {} datasets".format(len(datasets)))
for name, dataset in datasets.items():
    print(name, "columns: ", len(dataset.df.columns))
profile = ProfileReport(datasets['crowd-sourced-covid-19-testing-locations'].df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile
if os.path.isdir("./Uncover COVID Reports"):
    print("Directory already exists")
else:
    Path("./Uncover COVID Reports").mkdir(parents=True, exist_ok=False)
    print("created Folder 'Uncover COVID Reports'.")
print(os.getcwd())
min_threshold = 100

for name, dataset in datasets.items():
    if not os.path.isfile('./Uncover COVID Reports/'+ name +'.html'):
        reduce = True if len(dataset.df.columns) > min_threshold else False
        profile = ProfileReport(dataset.df, title='Pandas Profiling Report', html={'style':{'full_width':True}}, minimal=reduce)
        profile.to_file(output_file="./Uncover COVID Reports/"+name+".html")
        print("Created report for dataset: ", name)
os.chdir("./Uncover COVID Reports")
print(os.getcwd())
os.chdir("../")
print(os.getcwd())