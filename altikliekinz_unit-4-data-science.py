#Script that predicts the "speed" of cars based on the "temp_outside". 
import pandas as pd

data = pd.read_csv("../input/car-consume/measurements.csv")
data[["speed", "temp_outside"]].groupby("speed").mean()

