import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%%time

df = pd.read_csv("/kaggle/input/nzta-crash-analysis-system-cas/Crash_Analysis_System_CAS_data.csv")
df.columns
akl = df[df.region == "Auckland Region"]

akl.crashSeverity.value_counts()
akl.crashYear.value_counts().plot(kind="bar")
akl.weatherA.value_counts().plot(kind="bar")
akl.crashLocation1.value_counts().head(20)