
def int_to_hex(i):
    h = hex(i)
    s = str(h)
    o = s.split("0x")[1]
    if len(o) < 2:
        o = "0" + o
    return o


# print(int_to_hex(128))

def rgb_int_list_to_hex_string(rgb_list):
    output = []
    for i in rgb_list:
        output.append(int_to_hex(i))
    return "".join(output)


color_output = []
limit = 256
step = 8
for r in range(0, limit, step):
    for g in range(0, limit, step):
        for b in range(0, limit, step):
            row = "<div style=\"height:1em;width:1em;display:inline-block;background:#"
            row += rgb_int_list_to_hex_string([r, g, b])
            row += "\"></div>"
            color_output.append(row)

output_file = open("html_output_colors.html", "w")
output_file.write("\n".join(color_output))
output_file.close()
