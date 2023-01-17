import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
df=pd.read_csv('../input/kobe_data.csv')

df.head(2)
df.tail(2)
#df.info()
len(df)



df.drop(['game_event_id', 'game_id', 'lat','lon', 'minutes_remaining', 'period', 'playoffs',
       'season', 'seconds_remaining','team_id', 'team_name', 'game_date', 'matchup', 'opponent','shot_distance','shot_type','shot_zone_range','shot_id'], axis=1, inplace=True)

df.head(3)
df.tail(3)
shot_type_df = df.dropna()
len(df)
len(shot_type_df)
shot_type_df = shot_type_df[['action_type','combined_shot_type']]
shot_type_df.head(2)
shot_type_df.tail(2)
shot_group =shot_type_df.groupby(['combined_shot_type','action_type'])
type(shot_type_df)
type(shot_group)
type(shot_group.size())
shot_group.size()
shot_group = shot_group.size()
shot_group.index.get_level_values(0)
shot_group.index.get_level_values(1)
level0 = shot_group.index.get_level_values(0).tolist()
level1 = shot_group.index.get_level_values(1).tolist()
#shot_group.index.get_level_values('combined_shot_type')
#shot_group['Dunk'].index
#shot_group.size().index

shot_group_dict = shot_group.to_dict()
base_index = ['Bank Shot', 'Dunk', 'Hook Shot', 'Jump Shot', 'Layup', 'Tip Shot']

shot_group_dict_adv ={}
for index in base_index:
    shot_group_dict_adv[index] = {}
for index in shot_group_dict :
    shot_group_dict_adv[index[0]][index[1]] = shot_group[index]
#    print (index)
#shot_group_dict_adv
#shot_group_dict
import json

kobe_dic = {}
kobe_dic["name"]= "shot_selection"
kobe_dic["children"] = []

#for index in shot_group_dict_adv:
#    temp_dic = {"name":index,"children" : []}
#    kobe_dic["children"].append(temp_dic)
for index_0 in shot_group_dict_adv:
    temp_dic = {"name":index_0,"children" : []}
 #   kobe_dic["children"].append(temp_dic)
 #   kobe_dic["children"].append()
    for index_1 in shot_group_dict_adv[index_0]:
        temp_dic_2 = {"name":index_1,"size" : \
                      str(shot_group_dict_adv[index_0][index_1])}
        temp_dic['children'].append(temp_dic_2)
    kobe_dic["children"].append(temp_dic)    
#kobe_dic        
kobe_json_temp = json.dumps(kobe_dic)


with open('kobe_temp.json', 'w') as outfile:  
    json.dump(kobe_dic, outfile)
pd.read_json('kobe_temp.json').head()
#kobe_json
html_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>

.node {
  cursor: pointer;
}

.node:hover {
  stroke: #000;
  stroke-width: 1.5px;
}

.node--leaf {
  fill: white;
}

.label {
  font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;
  text-anchor: middle;
  text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;
}

.label,
.node--root,
.node--leaf {
  pointer-events: none;
}

</style>
<svg width="380" height="380"></svg>
"""
js_string="""
 require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });

  require(["d3"], function(d3) {

   console.log(d3);

var svg = d3.select("svg"),
    margin = 20,
    diameter = +svg.attr("width"),
    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")");

var color = d3.scaleSequential(d3.interpolateViridis)
    .domain([-4, 4]);

var pack = d3.pack()
    .size([diameter - margin, diameter - margin])
    .padding(2);

d3.json("kobe_temp.json", function(error, root) {
  if (error) throw error;

  root = d3.hierarchy(root)
      .sum(function(d) { return d.size; })
      .sort(function(a, b) { return b.value - a.value; });

  var focus = root,
      nodes = pack(root).descendants(),
      view;

  var circle = g.selectAll("circle")
    .data(nodes)
    .enter().append("circle")
      .attr("class", function(d) { return d.parent ? d.children ? "node" : "node node--leaf" : "node node--root"; })
      .style("fill", function(d) { return d.children ? color(d.depth) : null; })
      .on("click", function(d) { if (focus !== d) zoom(d), d3.event.stopPropagation(); });

  var text = g.selectAll("text")
    .data(nodes)
    .enter().append("text")
      .attr("class", "label")
      .style("fill-opacity", function(d) { return d.parent === root ? 1 : 0; })
      .style("display", function(d) { return d.parent === root ? "inline" : "none"; })
      .text(function(d) { return d.data.name; });

  var node = g.selectAll("circle,text");

  svg
      .style("background", color(-1))
      .on("click", function() { zoom(root); });

  zoomTo([root.x, root.y, root.r * 2 + margin]);

  function zoom(d) {
    var focus0 = focus; focus = d;

    var transition = d3.transition()
        .duration(d3.event.altKey ? 7500 : 750)
        .tween("zoom", function(d) {
          var i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2 + margin]);
          return function(t) { zoomTo(i(t)); };
        });

    transition.selectAll("text")
      .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
        .style("fill-opacity", function(d) { return d.parent === focus ? 1 : 0; })
        .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
        .on("end", function(d) { if (d.parent !== focus) this.style.display = "none"; });
  }

  function zoomTo(v) {
    var k = diameter / v[2]; view = v;
    node.attr("transform", function(d) { return "translate(" + (d.x - v[0]) * k + "," + (d.y - v[1]) * k + ")"; });
    circle.attr("r", function(d) { return d.r * k; });
  }
});
  });
 """
import IPython.display
from IPython.core.display import display, HTML, Javascript
h = display(HTML(html_string))
j = IPython.display.Javascript(js_string)
IPython.display.display_javascript(j)
shoot_percentage = df.dropna()[['combined_shot_type','shot_made_flag']].groupby(['combined_shot_type']).mean().sort_values(by='shot_made_flag',ascending=1)
shoot_percentage
shoot_percentage.index
shoot_percentage.columns
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x=shoot_percentage.index,y=shoot_percentage.shot_made_flag)
#plt.xticks(rotation=45)
plt.show()
df.head(2)
df.tail(2)
df['shot_zone_combined'] = df[['shot_zone_area','shot_zone_basic']].apply(lambda x: ' '.join(x), axis=1)
df.head(2)
df.tail(2)
shot_zone_map = df.groupby(['shot_zone_combined']).mean()['shot_made_flag']
shot_zone_map
df['avg_by_shot_zone_combined'] = df['shot_zone_combined'].map(shot_zone_map)
df.head(2)
df.tail(2)
df = df[df['loc_y']>=0]
fig = plt.figure(figsize=(10,8))
plt.scatter(df['loc_x'],df['loc_y'],c=df['avg_by_shot_zone_combined'])
plt.xlim(280,-280)
plt.ylim(-10,450)
plt.colorbar()
plt.title('Kobe Bryant\'s shot accuracy by Shot Zone')

plt.show()
#plt.savefig('Kobe Bryants shot accuracy by zone.png')



