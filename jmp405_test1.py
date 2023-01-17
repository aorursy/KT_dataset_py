import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train_users_2.csv')
train.head()
test = pd.read_csv('../input/test_users.csv')
test.head()
sessions = pd.read_csv('../input/sessions.csv')
sessions.head()
countries = pd.read_csv('../input/countries.csv')
countries.head()
age_gender = pd.read_csv('../input/age_gender_bkts.csv')
age_gender.head()
