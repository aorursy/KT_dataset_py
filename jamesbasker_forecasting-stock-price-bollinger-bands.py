import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df = pd.read_csv("../input/GOOG_032016_032017.csv")
print(df)
rolling_mean = df['Close'].rolling(20).mean()
rolling_std = df['Close'].rolling(20).std()
bollinger_upper_band = rolling_mean + rolling_std * 2
bollinger_lower_band = rolling_mean - rolling_std * 2
import matplotlib.pyplot as plt

ax = df['Close'].plot(title="Bollinger Bands", fontsize=12)

rolling_average = rolling_mean

rolling_average.plot(label='Rolling mean', ax=ax)

bollinger_upper_band.plot(label='Bollinger upper band', ax=ax)

bollinger_lower_band.plot(label='Bollinger lower band', ax=ax)

plt.show()