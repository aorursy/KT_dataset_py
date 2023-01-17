import bar_chart_race as bcr
!pip install bar-chart-race

df= bcr.load_dataset("covid19_tutorial")
df
bcr.bar_chart_race(df, orientation="v")
bcr.bar_chart_race(df, orientation="v", sort="asc")
bcr.bar_chart_race(df, orientation="v", sort="asc", steps_per_period=20, period_length=200
                   )
bcr.bar_chart_race(df, orientation="v", sort="asc", interpolate_period=True)
bcr.bar_chart_race(df, orientation="v", sort="asc", figsize=(5,3), title="Covid19 Deaths by Country", interpolate_period=True)