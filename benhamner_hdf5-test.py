import h5py
import pandas as pd

iris = pd.read_csv("../input/Iris.csv")

iris.to_hdf("iris.h5", "iris", index=False)

with h5py.File("iris.h5", "r") as f:
      for key in f["iris"].keys():
            print("**%s**" % key)
            print(f["iris"][key])
      print(f["iris"]["axis0"][:])
      print(f["iris"]["axis1"][:])
      print(f["iris"]["block0_items"][:])
      print(f["iris"]["block0_values"][:])
        
      
