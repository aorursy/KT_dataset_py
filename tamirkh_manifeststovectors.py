import os

import xml.etree.ElementTree as ET



data_address = "/kaggle/input/apk-manifests" # address of apks

all_used_permission = set() # the set which will contain all the permissions

counter = 0 # count all the cases where a permissions was already known



for filename in os.listdir(data_address): # iterate over the data

    root = ET.parse(data_address + "/" + filename).getroot()

    permissions = root.findall("uses-permission") # extract all the elements with "uses-permission"

    for perm in permissions:

        for att in perm.attrib:

            if perm.attrib[att] in all_used_permission:

                counter+=1

            all_used_permission.add(perm.attrib[att]) # add the permission to the permission set



path, dirs, files = next(os.walk(data_address)) # check number of manifests in the dataset

            

print("number of manifests: " + str(len(files)))

print("number of distinct permissions: " + str(len(all_used_permission)))

print("number caases a permissions was already seen: " + str(counter))
enumerated_permissions = {} # the dictionary will convert manifest to its id

permission_id = 0



for permission in all_used_permission:

    enumerated_permissions[permission] = permission_id

    permission_id+=1
def manifestToVector(enumerated_permissions, manifest_name):

    result = [0]*121

    

    root = ET.parse(data_address + "/" + filename).getroot()

    permissions = root.findall("uses-permission") # extract all the elements with "uses-permission"

    for perm in permissions:

        for att in perm.attrib:

            result[enumerated_permissions[perm.attrib[att]]] = 1

                   

    return result
import numpy as np



permissions_table = array = np.arange(121) # a temp row, will be deleted later



for filename in os.listdir(data_address): # iterate over the data

    manifest_as_vector = np.asarray(manifestToVector(enumerated_permissions, filename)) # convert manifest to array

    permissions_table = np.vstack([permissions_table, manifest_as_vector]) # add the vector as a new row



permissions_table = np.delete(permissions_table, (0), axis=0) # delete first row

print(permissions_table[59])