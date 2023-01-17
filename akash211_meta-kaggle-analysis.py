import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas_profiling as pp
Teams = pd.read_csv('../input/Teams.csv',low_memory=False)
Users = pd.read_csv('../input/Users.csv',low_memory=False)
Datasets = pd.read_csv('../input/Datasets.csv',low_memory=False)
Forums = pd.read_csv('../input/Forums.csv',low_memory=False)
Submissions = pd.read_csv('../input/Submissions.csv',low_memory=False)
KernelVotes = pd.read_csv('../input/KernelVotes.csv',low_memory=False)
DatasetVersions = pd.read_csv('../input/DatasetVersions.csv',low_memory=False)

pp.ProfileReport(Teams)


pp.ProfileReport(Users)
pp.ProfileReport(Datasets)
pp.ProfileReport(Forums)
pp.ProfileReport(Submissions)
pp.ProfileReport(KernelVotes)
pp.ProfileReport(DatasetVersions)

