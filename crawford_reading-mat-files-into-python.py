import scipy.io
# Import to a python dictionary

mat = scipy.io.loadmat('../input/Dataset_PerCom18_STL/Dataset_PerCom18_STL/cross_opp.mat')
# Look at the dictionary items

mat.items()
# Print the data

mat["data_opp"]