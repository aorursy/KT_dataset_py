output_path = "file.txt"

temp_path = "../file.txt"

kilobyte = 1024

with open(output_path, "w") as f:

    for x in range(1024*1024*2):

        f.write("a" * kilobyte)
kilobyte = 1024

with open(temp_path, "w") as f:

    for x in range(1024*1024*2):

        f.write("a" * kilobyte)
import gc

gc.collect()
!ls -l ..

!du .
!du ..