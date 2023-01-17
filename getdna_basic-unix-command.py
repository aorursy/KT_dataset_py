!pwd
!mkdir "A random Folder"
!ls
!cd "./A random Folder"; pwd
print("Current Directory")

!pwd

print("After cd..")

!cd ..; pwd
!cd /; pwd
# คำสั่งเขียนไฟล์ทั่วๆ ไป

# Reference: https://www.w3schools.com/python/python_file_write.asp

f = open("sample.txt", "w")

f.write("This is a sample file!")

f.close()
print("!ls")

!ls

print("!cat sample.txt")

!cat sample.txt
!cp sample.txt sample2.txt

!ls
print("!ls")

!ls

print("!rm sample.txt")

!rm sample.txt

print("!ls")

!ls