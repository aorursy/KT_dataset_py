file_name = "log.txt"
APPEND = "a"
WRITE = "w"
READ = "r"

file_handle = open(file_name, WRITE)
file_handle.write("hello world\n")
file_handle.write("goodbye world\n")
file_handle.close()

file_handle = open(file_name, APPEND)
file_handle.write("hello world!!!\n")
file_handle.write("goodbye world!!!\n")
file_handle.close()