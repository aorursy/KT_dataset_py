import pandas as pd

import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from tqdm import tqdm
X_byte = pd.read_csv("../input/byte_final_features.csv")
byte_entropy_filesize = X_byte[["Id", "entropy", "fsize"]]

byte_entropy_filesize = byte_entropy_filesize.set_index("Id")
poly = PolynomialFeatures(3)

x_byte_poly = poly.fit_transform(byte_entropy_filesize)
byte_entropy_filesize.head()
x_byte_poly = x_byte_poly[:, 1:]
x_byte_df = pd.DataFrame(x_byte_poly, columns = ["byte_fe"+str(i) for i in range(x_byte_poly.shape[1])] )
byte_entropy_filesize = byte_entropy_filesize.reset_index()

x_byte_df["Id"] = byte_entropy_filesize["Id"]
x_byte_df.head()
X_asm = pd.read_csv("../input/asm_reduced_final.csv", index_col = 0) #index_col removes unnamed column

X_asm.head()
asm_file_size = pd.read_csv("../input/asm_file_size.csv")
X_asm = X_asm.merge(asm_file_size, on="Id")
#Very important

#data = X_asm.merge(y_asm, on="Id")

#data.head()

data_asm_byte_final = X_asm.merge(x_byte_df, on="Id")

data_asm_byte_final.head()
data_asm_byte_final.shape
data_asm_byte_final.to_csv("data_asm_byte_final.csv")

#data_asm_byte_final will be the final data to be used for modelling.