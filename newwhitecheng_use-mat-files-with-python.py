import scipy.io as sio 
# Import to a python dictionary
file = sio.loadmat('../input/Dataset_PerCom18_STL/Dataset_PerCom18_STL/cross_opp.mat')
# Look at the dictionary items
# you have item data_opp
file.items()
# Print the data
file["data_opp"]