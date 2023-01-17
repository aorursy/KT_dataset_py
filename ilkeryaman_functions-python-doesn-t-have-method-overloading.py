def method_to_overload(name):
    print(name)
def method_to_overload(some_id, name):
    print(some_id, name)
method_to_overload("ilker")

method_to_overload(1, "ilker")