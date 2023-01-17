
class ContainsStuff:

    def __init__(self, list_of_stuff):
        self.list_of_stuff = list_of_stuff

    def __str__(self):
        output = []
        for item in self.list_of_stuff:
            output.append(str(item))
        return ", ".join(output)


my_list = [123, 456, 234]
cs = ContainsStuff(my_list)

print(cs)

