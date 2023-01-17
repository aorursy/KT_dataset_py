def greeting(name):
    print("Hello", name)
greeting("ilker")

myName = None
greeting(myName)
def greeting2(name="ilker"):
    greeting(name)
greeting2(myName)
greeting2()

emptyString = ""
greeting2(emptyString)
def details(name="ilker", surname="yaman", phone="5321112233"):
    print("name: {}, surname: {}, phone: {}".format(name, surname, phone))
details()
details(surname="yemen")
def total(*a):
    ttl = 0
    for i in a:
        ttl += i
    return ttl


print(total(5, 4, 3))
def print_local_glb():
    glb = 2
    print(glb)
def print_glb():
    global glb  # use global glb
    print(glb)
glb = "10"
print_local_glb()

print(glb)

print_glb()
def update_glb():
    global glb  # use global glb
    glb = 4
update_glb()

print(glb)