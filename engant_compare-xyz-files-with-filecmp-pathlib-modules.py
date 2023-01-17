# https://docs.python.org/3.6/library/filecmp.html
# https://docs.python.org/3/library/pathlib.html

import filecmp

xyz_folders = map(str, range(1, 2400))
xyz_files = [folder+"/geometry.xyz" for folder in xyz_folders]

match, mismatch, errors = filecmp.cmpfiles("../input/train", "../input/train", 
                                           xyz_files, shallow=False)
print("match", match)
print("mismatch", mismatch)
print("errors", errors)
from pathlib import Path
all_xyz = list(Path('../input').glob('**/*.xyz'))
#all_xyz = sorted(Path('../input').glob('**/*.xyz'))
cnt_xyz = len(all_xyz)
print("total # geometry files:", cnt_xyz)

print(all_xyz[:12])
print('*'*21)
print("tail:\n", all_xyz[-12:])
# str(all_xyz[0])
similar_files = []
for file1 in all_xyz:
    for file2 in all_xyz:
        cmp_flag = filecmp.cmp(str(file1), str(file2), shallow=False)
        if cmp_flag and str(file1) != str(file2):
            ## add file string of lesser subfolder integer
            ## to the front of our result string
            int_sub1 = int(str(file1).split('/')[3])
            int_sub2 = int(str(file2).split('/')[3])
            if int_sub1 <= int_sub2:
                # equal sign not needed
                similar_files.append(f"{file1}, {file2}")
            else:
                #int_sub1 > int_sub2:
                similar_files.append(f"{file2}, {file1}")
                
## get rid of duplicates
similar_files = set(similar_files)
print(similar_files)