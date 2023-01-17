

data_list = ["123", "345", "234", "111"]
write_me = "<BR>".join(data_list)
output_file = open("html_output.html", "w")
output_file.write(write_me)
output_file.close()

columns_list = ["AAA", "BBB", "CCC", "DDD"]

colors = ["red", "green", "blue", "black"]

output = "<table>"

for index in range(len(columns_list)):
    column_name = columns_list[index]
    value = data_list[index]
    color = colors[index]

    output += "<tr><td>"
    output += column_name
    output += "</td><td><div style=\"width:"
    output += value
    output += "px;height:1em;display:inline-block;background:"
    output += color
    output += "\"></div>("
    output += value
    output += ")</td></tr>"

output += "</table>"

output_file = open("html_output_bars.html", "w")
output_file.write(output)
output_file.close()
