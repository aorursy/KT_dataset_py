import hashlib
salt = ""
text= "password" + salt
m = hashlib.md5()
m.update(text.encode())

print(m.hexdigest())
