import matplotlib.pyplot as plt
from part1_cleaning import *

def articles_per_month(df, l):
    df["Headlines"].groupby([df["Date"].dt.year, df["Date"].dt.month]).count().plot(kind="bar")
    plt.title("Number of Articles per Month (%s)" %l)
df1, df2, df3 = get_clean_data()
articles_per_month(df1, "CNBC")
articles_per_month(df2, "Reuters")
articles_per_month(df3, "Guardian")
