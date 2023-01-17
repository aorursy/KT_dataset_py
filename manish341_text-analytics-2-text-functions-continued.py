## File Operations
# Open a file
# f = open(filename, mode) <-- format for open function.

f = open('../input/test.txt', 'r')  # The default mode is 'rt' (open for reading text).
                           # 'r' = read mode,     'x' : create a new file,
                           # 'w' = write mode,    'a' = open for writing
                           # 'b' = binary mode,   't' = text mode
                           # '+' = update mode,   'U' = Universal new line mode (deprecated)

# Reading file line by line
f.readline()    # Read the first line
# Reading next line? Run the same code again.
f.readline()
#Set the cursor to start of file
f.seek(0) 
print(f.readline())
#  Read data in chunks of 10 characters
f.seek(0) 
print(f.readline(10))     # Read first 10 bytes
print(f.readline(50))     # Read next few bytes
print(f.readline(15))     # Read bytes from another line
# seek to start and read all lines as a list element
f.seek(0)
print(f.readlines())

# Try reading only 40 characters from start
f.seek(0)
print(f.readlines(40))
# Strip out the next line character (or any white space character) from output
f.seek(0)
lines = f.readlines()
[line.rstrip() for line in lines]
# Removing white spaces from texts
txt1 = " This sentence has white spaces at both sides. "
txt2 = "This sentence has white spaces at right side only. "
txt3 = " This sentence has white spaces on the left side."

# Removing white spaces from both sides
print(txt1.strip())
print("Length before: " + str(len(txt1)) + ", and after: " + str(len(txt1.strip())) + " \n")

# Removing white spaces from right side
print(txt2.rstrip())
print("Length before: " + str(len(txt2)) + ", and after: " + str(len(txt2.rstrip())) + " \n")

# Removing white spaces from left side
print(txt3.lstrip())
print("Length before: " + str(len(txt3)) + ", and after: " + str(len(txt3.lstrip())) + " \n")
# Reading the full file
f.seek(0)
txt1 = f.read()
txt1
# split the file into list by lines (by '\n')
txt1.splitlines()      # It will return a list of all lines
# We can also write the data back to a back
f.write("We are writing new data to the text file.")
f.close()      # Close the current file connection

# Open it back in write mode
# f = open('test_new.txt', 'w')
# f.write("This is a new line we are inserting to the data.")
# f.close()
f.closed