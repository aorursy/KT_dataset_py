# alternative before classes were invented
def make_category(name):
    s = {}
    s["name"] = name
    s["menu_items"] = []
    s["add_menu_item"] = lambda mi: s["menu_items"].append(mi)
    return s    

c = make_category("Dessert")

c["add_menu_item"]("Donut")

print(c["name"])

print(c["menu_items"])

