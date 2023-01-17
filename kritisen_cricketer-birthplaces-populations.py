# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.core.display import display, HTML, Javascript
from string import Template
import IPython.display
import json
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv('/kaggle/input/indian-cricketer-origins/data/odi-cricketers.csv')
df1.head()
df2 = pd.read_csv('/kaggle/input/indian-cricketer-origins/data/Population.csv')
df2.head()
result = df1.to_json(orient="records")
with open('cricketers.json', 'w') as outfile:  
    json.dump(result, outfile)
popn = df2.to_json(orient="records")
with open('population.json', 'w') as outfile:  
    json.dump(popn, outfile)
html_barchart1_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>
    .wrapper {
      position: relative;
    }
    .tooltip {
      position: absolute;
      left: 0;
      top: 0;
      width: auto;
      height: auto;
      background-color: white;
      border: solid;
      border-width: 1px;
      border-radius: 5px;
      padding: 10px;
      font-size: 10px;
      font-family: courier;
      visibility: hidden;
      opacity: 1;
    }
</style>
<div class='wrapper'>
    <div class='chart'></div>
    <div class='tooltip'></div>
</div>
"""
js_barchart1_string = """
 require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });

  require(["d3"], function(d3) {
     const margin = {top: 30, right: 30, bottom: 60, left: 60};
     const width = 500 - margin.left - margin.right;
     const height = 300 - margin.top - margin.bottom;

     const svg = d3.select(".chart")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");
              

     d3.json('cricketers.json', function(error, players) {
         
         if (error) throw error;
         ready(JSON.parse(players))
     })

      function ready(players) {
      
        players.forEach(d=>{
          d.classification = classification(+d.population)
        })

        let playersGrp = d3.nest()
          .key(d=>d.classification)
          .rollup(function(leaves) { return leaves.length; })
          .entries(players)

        let playersValues = d3.nest()
          .key(d=>d.classification)
          .entries(players)

        // Initialize the plot
        update(playersGrp, playersValues, "Birthplace population", "# ODI cricketers", d3.formatPrefix(".0", 0), ["Player_name", null, 'List of players'])

      }

        function classification(d){
          if(d > 6000000){
            return '> 6 million'
          } else if(d > 1000000 & d <= 6000000){
            return '1-6 million'
          } else if(d <= 1000000){
            return '< 1 million'
          }
        }

        function update(data, data1, xLabel, yLabel, yFormat, tooltip_attrs) {

          const sortingArr = ['< 1 million', '1-6 million', '> 6 million']
          const categories = data.map(function(d) { return d.key; })

          const categoriesSorted = categories.sort((a, b) => {
            return sortingArr.indexOf(a) - sortingArr.indexOf(b);
          })

          let colorScale = d3.scaleOrdinal()
            .range(['green', 'blue', 'orange'])
            .domain(categoriesSorted)

          let colorAccessor = (d) => colorScale(d.key)

          const x = d3.scaleBand()
            .range([ 0, width ])
            .domain(categoriesSorted)
            .padding(0.2);

          svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))

          svg.append("text")
            .attr("transform", `translate(${width/2}, ${height+40})`)
            .attr('text-anchor', 'middle')
            .attr("font-size", '11px')
            .text(xLabel)

          const y = d3.scaleLinear()
            .domain([0, d3.max(data, d=>d.value)])
            .range([ height, 0]);

          svg.append("g")
            .attr("class", "myYaxis")
            .call(d3.axisLeft(y).tickFormat(yFormat));

          svg.append("text")
            .attr("transform", `translate(-40,${height/2})rotate(-90)`)
            .attr('text-anchor', 'middle')
            .attr("font-size", '11px')
            .text(yLabel)

          const bar = svg.selectAll("rect")
            .data(data)

          bar
            .enter()
            .append("rect")
            .merge(bar)
              .attr("x", function(d) { return x(d.key); })
              .attr("y", function(d) { return y(d.value); })
              .attr("width", x.bandwidth())
              .attr("height", function(d) { return height - y(d.value); })
              .attr("fill", colorAccessor)
              .attr('cursor', 'pointer')
              .on("mouseover", function(d) { 
                 let items = data1.find(el=>el.key === d.key).values.map(el=>el[tooltip_attrs[0]])
                 let items_1 = data1.find(el=>el.key === d.key).values.map(el=>el[tooltip_attrs[1]])
                 var keywordList = "<ul>";
                  for(var i = 0; i < items.length; i++){
                    tooltip_attrs[1] ? keywordList += "<li>" + items[i] + " ("+ d3.format(".2s")(items_1[i]) + " people)</li>" :  keywordList += "<li>" + items[i] + "</li>" 
                  }
                  keywordList += "</ul>";
                 d3.select('.tooltip')
                   .style('top', `${d3.select(this).attr('y')+30}px`)
                   .style('left', `${+d3.select(this).attr('x')+x.bandwidth()/2+120}px`)
                   .style('visibility', 'visible')
                   .html("<b>" + tooltip_attrs[2] + ": </b>" + keywordList)
              })
              .on("mouseout", function() { 
                 d3.select('.tooltip')
                   .style('top', '0px')
                   .style('left', '0px')
                   .style('visibility', 'hidden')
                   .html("")
              })

        }
    })
"""
h_barchart1 = display(HTML(html_barchart1_string))
j_barchart1 = IPython.display.Javascript(js_barchart1_string)

IPython.display.display_javascript(j_barchart1)
html_barchart2_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>
    .wrapper2 {
      position: relative;
    }
    .tooltip2 {
      position: absolute;
      left: 0;
      top: 0;
      width: auto;
      height: auto;
      background-color: white;
      border: solid;
      border-width: 1px;
      border-radius: 5px;
      padding: 10px;
      font-size: 10px;
      font-family: courier;
      visibility: hidden;
      opacity: 1;
    }
</style>
<div class='wrapper2'>
    <div class='chart2'></div>
    <div class='tooltip2'></div>
</div>
"""
js_barchart2_string = """
 require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });

  require(["d3"], function(d3) {
     const margin = {top: 30, right: 30, bottom: 60, left: 60};
     const width = 500 - margin.left - margin.right;
     const height = 300 - margin.top - margin.bottom;

     const svg = d3.select(".chart2")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");
              

     d3.json('population.json', function(error, data) {
         
         if (error) throw error;
         ready(JSON.parse(data))
     })

      function ready(popn) {
        popn.forEach(d=>{
          d.classification = classification(+d.population)
        })

        let popnGrp = d3.nest()
          .key(d=>d.classification)
          .rollup(function(leaves) { return d3.sum(leaves, d=>+d.population) })
          .entries(popn)

        let popnValues = d3.nest()
          .key(d=>d.classification)
          .entries(popn)

        // Initialize the plot
        console.log(popnGrp)
        update(popnGrp, popnValues, "Birthplace population", "Sum of populations of birthplaces in group", d3.formatPrefix(".0", 1e6), ["Birthplace", 'population', 'List of birthplaces'])

      }

        function classification(d){
          if(d > 6000000){
            return '> 6 million'
          } else if(d > 1000000 & d <= 6000000){
            return '1-6 million'
          } else if(d <= 1000000){
            return '< 1 million'
          }
        }

        function update(data, data1, xLabel, yLabel, yFormat, tooltip_attrs) {

          const sortingArr = ['< 1 million', '1-6 million', '> 6 million']
          const categories = data.map(function(d) { return d.key; })

          const categoriesSorted = categories.sort((a, b) => {
            return sortingArr.indexOf(a) - sortingArr.indexOf(b);
          })

          let colorScale = d3.scaleOrdinal()
            .range(['green', 'blue', 'orange'])
            .domain(categoriesSorted)

          let colorAccessor = (d) => colorScale(d.key)

          const x = d3.scaleBand()
            .range([ 0, width ])
            .domain(categoriesSorted)
            .padding(0.2);

          svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))

          svg.append("text")
            .attr("transform", `translate(${width/2}, ${height+40})`)
            .attr('text-anchor', 'middle')
            .attr("font-size", '11px')
            .text(xLabel)

          const y = d3.scaleLinear()
            .domain([0, d3.max(data, d=>d.value)])
            .range([ height, 0]);

          svg.append("g")
            .attr("class", "myYaxis")
            .call(d3.axisLeft(y).tickFormat(yFormat));

          svg.append("text")
            .attr("transform", `translate(-40,${height/2})rotate(-90)`)
            .attr('text-anchor', 'middle')
            .attr("font-size", '11px')
            .text(yLabel)

          const bar = svg.selectAll("rect")
            .data(data)

          bar
            .enter()
            .append("rect")
            .merge(bar)
              .attr("x", function(d) { return x(d.key); })
              .attr("y", function(d) { return y(d.value); })
              .attr("width", x.bandwidth())
              .attr("height", function(d) { return height - y(d.value); })
              .attr("fill", colorAccessor)
              .attr('cursor', 'pointer')
              .on("mouseover", function(d) { 
                 let items = data1.find(el=>el.key === d.key).values.map(el=>el[tooltip_attrs[0]])
                 let items_1 = data1.find(el=>el.key === d.key).values.map(el=>el[tooltip_attrs[1]])
                 var keywordList = "<ul>";
                  for(var i = 0; i < items.length; i++){
                    tooltip_attrs[1] ? keywordList += "<li>" + items[i] + " ("+ d3.format(".2s")(items_1[i]) + " people)</li>" :  keywordList += "<li>" + items[i] + "</li>" 
                  }
                  keywordList += "</ul>";
                 d3.select('.tooltip2')
                   .style('top', `${d3.select(this).attr('y')+30}px`)
                   .style('left', `${+d3.select(this).attr('x')+x.bandwidth()/2+120}px`)
                   .style('visibility', 'visible')
                   .html("<b>" + tooltip_attrs[2] + ": </b>" + keywordList)
              })
              .on("mouseout", function() { 
                 d3.select('.tooltip')
                   .style('top', '0px')
                   .style('left', '0px')
                   .style('visibility', 'hidden')
                   .html("")
              })

        }
    })
"""
h_barchart2 = display(HTML(html_barchart2_string))
j_barchart2 = IPython.display.Javascript(js_barchart2_string)

IPython.display.display_javascript(j_barchart2)
html_map_string = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>
    .wrapper_map {
      position: relative;
    }
    .chart_map {
      position: relative;
    }
    .tooltip {
      display: flex;
      align-items: center;
      justify-content: center;
      position: absolute;
      left: 0px;
      top: 0px;
      width: 300px;
      height: 140px;
      opacity: 0;
      background-color: white;
      border: solid;
      border-width: 1px;
      border-radius: 5px;
      padding: 4px;
      font-family: courier;
    }
    .tooltip p {
      font-size: 12px;
      margin: 3px 0px;
    }
    .tooltip h3 {
      margin: 0px 0px 10px 0px;
    }
    .tooltip a {
      font-size: 12px;
      color: navy;
    }
    .tooltip span {
      font-weight: bold;
    }
    text {
      font-family: courier;
      font-size: 11px;
      fill: black;
    }
    .buttons {
      position: absolute;
      bottom: 45px;
      left: 92px;
    }
</style>
<div class='wrapper_map'>
  <div class='chart_map'></div>
  <div class='buttons'>
    <button id="zoom-in">+</button>
    <button id="zoom-out">-</button>
  </div>
</div>
"""
js_map_string = """
 require.config({
    paths: {
        d3: "https://d3js.org/d3.v5.min"
     }
 });

  require(["d3"], function(d3) {


      Promise.all([
        d3.json('india_district_filtered.geojson'),
        d3.json('Indian_States.geojson'),
        d3.json ('cricketers.json')
      ]).then(
        d => ready(null, d[0], d[1], d[2])
      );

      function ready(error, districts, states, players) {
        drawMap(districts, states, players, 'bubble')
      }


        let clicked = false

        const svg = d3.select('.chart_map')
          .append('svg')

        const g = svg.append('g')
          .attr('class', 'map')
          .attr('transform', 'translate(0, 0)')

        function Zoom(initZoom, svg) {

          const zoom = d3.zoom()
            .scaleExtent([1, 20])
            .on('zoom', zoomed);

          if (initZoom == 2) { // delhi
            g.attr('transform', `translate(-500.3975825439707,-80.70595060723429) scale(4)`)
          } if (initZoom == 3){ //chennai - bangalore
            g.attr('transform', `translate(-500.3975825439707,-1000.70595060723429) scale(4)`)
          } 
          svg.call(zoom)

          function zoomed() {

            g.attr('transform', d3.event.transform)

            //g.selectAll(".player-marker")
              //.attr('r', function(d) { return d3.select(this).attr("r")/d3.event.transform.k })
              //.attr("stroke-width", 1/d3.event.transform.k)

          }

          d3.select('#zoom-in').on('click', function() {
            zoom.scaleBy(svg.transition().duration(750), 1.3);
          });

          d3.select('#zoom-out').on('click', function() {
            zoom.scaleBy(svg.transition().duration(750), 1 / 1.3);
          });

        }

        function tooltip(svg) {

          // create a tooltip
          let Tooltip = d3.select(".chart")
            .append("div")
            .attr("class", "tooltip")

          let mouseover = function(d) {
            clicked = false
            Tooltip
              .html(
                "<div style='padding: 10px'><img width='90px' src=" + d.Player_image + "></div><div>" + 
                "<h3>" + d.Player_name + "</h3><p>" + 
                "State: <span>" + d.State + "</span></p><p>" + 
                "Birthplace: <span>" + d.Birthplace + "</span></p><p>" + 
                "Birthplace Pop.: <span>" + d.population + "</span></p><p>" + 
                "Role: <span>" + d.Major_contribution + "</span></p><p>" +         
                "Number of matches: <span>" + d.Number_of_ODIs_played + "</span></p>" + 
                "<a href=" + d.Player_url + " target='_blank'>Cricinfo profile</a>"
              )
              .style("left", (d3.event.clientX + 25).toString() + "px")
              .style("top", (d3.event.clientY - 70).toString() + "px")
              //.style("left", (projection([+d.Longitude, +d.Latitude])[0] + 25).toString() + "px")
              //.style("top", (projection([+d.Longitude, +d.Latitude])[1] - 70).toString() + "px")
              .style("opacity", 1)

            svg.selectAll('#marker-' + d.Player_name.replace(/\s/g, ""))
              .attr("fill-opacity", 1)
              .lower()

          }

          let mouseleave = function(d) {

            if(clicked){ // prevent tooltip from hiding if marker is clicked
              Tooltip.style("opacity", 1)
            } else {
              Tooltip.style("opacity", 0)
            }

            svg.selectAll('circle.player-marker')
              .attr("fill-opacity", .3)
          }

          return {mouseover, mouseleave}
        }

        function drawMap(districts, states, players, type, zoomed) {

          const margin = {top: 0, right: 0, bottom: 0, left: 0};
          const width = zoomed ? 400 : 850 - margin.left - margin.right;
          const height = width*0.85 - margin.top - margin.bottom;

          svg.attr('width', width)
            .attr('height', height)

          const projection = d3.geoMercator()
            .center([78.9629, 20.5937])
            .scale(width*1.55) 
            .translate([ width/2, height/2 + 20 ])

          const path = d3.geoPath().projection(projection);

          let map = g.append('g')

          let statesG = map.selectAll('g.state')
            .data(states.features)
            .enter().append("g")
            .attr("class", 'state')

          statesG 
            .append('path')
              .attr('class', 'state-path')
              .attr('d', path)
              .style('fill', 'lightgray')
              .style('stroke', 'none')
              .style('stroke-width', 0)
              .style('opacity', 1)

          const highlightedStates = ["Delhi", "Kolkata", "Chennai", "Bangalore Urban", "Hyderabad", "Greater Bombay"]

          let statesAccessor = (d) => highlightedStates.indexOf(d.properties.NAME_2) !== -1

          let data = districts.features.filter(statesAccessor)

          let districtsG = map.selectAll('g.district')
            .data(data)
            .enter().append("g")
            .attr("class", 'district')

          districtsG 
            .append('path')
              .attr('class', 'district-path')
              .attr('d', path)
              .style('fill', 'black')
              .style('stroke', 'none')
              .style('stroke-width', 0)
              .style('opacity', 1)

          // only append labels for selected districts
          districtsG 
            .append('text')
              .attr('class', 'district-text')
              .attr("transform", function (d) { return "translate(" + path.centroid(d) + ")"; })
              .attr('dy', '-1.25em')
              .attr("text-anchor", "middle")
              .attr("alignment-baseline", "central")
              .attr("font-size", zoomed ? '4px' : "10px")
              .text(function (d) { 
                if(d.properties.NAME_2 === "Bangalore Urban"){
                  return "Bangalore"
                } else if(d.properties.NAME_2 === "Greater Bombay"){
                  return "Mumbai"
                } else {
                  return d.properties.NAME_2;
                }
              })

            players.forEach(d=>{
              d.classification = classification(+d.population)
            })


            if(type==='adjusted_delhi'){

              Zoom(2, svg) 
              drawMarkersAdj(players, g, projection)

            } else if(type==='adjusted_chennai_bangalore'){

              Zoom(3, svg) 
              drawMarkersAdj(players, g, projection)

            } else if(type==='bubble'){

              const title = svg.append('g')
                .attr('class', 'title')
                .attr('transform', 'translate(92, 30)')

              title.append("text")
                .text('Indian Cricketers')
                .style('font-size', '2em')

              Zoom(1, svg) 

              drawMarkers(players, g, projection)
            }

        }

        function drawMarkers(data, svg, projection) {

          let {mouseover, mouseleave} = tooltip(svg)
          let sortingArr = ['> 6 million', '1-6 million', '< 1 million']
          let roles = data.map(d=>d.classification).filter(onlyUnique)

          roles = roles.sort((a, b) => {
            return sortingArr.indexOf(a) - sortingArr.indexOf(b);
          })

          let colorScale = d3.scaleOrdinal()
            .range(['orange', 'blue', 'green'])
            .domain(roles)

          let radiusScale = d3.scaleSqrt()
            .range([0, 20])
            .domain([0, d3.max(data, d => +d.Number_of_ODIs_played)])

          let colorAccessor = (d) => colorScale(d.classification)
          let radiusAccessor = (d) => radiusScale(+d.Number_of_ODIs_played)

          let markers = svg.append('g')

          let playerG = markers.selectAll('circle.player-marker')
            .data(data)
            .enter().append('circle')
              .attr('class', 'player-marker')
              .attr('id', d => 'marker-' + d.Player_name.replace(/\s/g, ""))
              .attr("cx", function (d) { return projection([+d["Longitude (ORIG)"], +d["Latitude (ORIG)"]])[0] })
              .attr("cy", function (d) { return projection([+d["Longitude (ORIG)"], +d["Latitude (ORIG)"]])[1] })
              .attr('r', radiusAccessor)
              .attr("fill-opacity", .3)
              .attr('fill', colorAccessor)
              .attr('stroke', colorAccessor)
              .attr("stroke-width", 1)
              .style('cursor', 'pointer')
              .on("click", function(){
                mouseover
                clicked = !clicked
              })
              .on("mouseover", mouseover)
              .on("mouseleave", mouseleave)

          drawLegend(roles, colorScale, d3.select('svg')) 
          drawRadiusLegend(radiusScale, d3.select('svg'))

        }

        function drawMarkersAdj(data, svg, projection) {

          let {mouseover, mouseleave} = tooltip(svg)
          let roles = data.map(d=>d.classification).filter(onlyUnique)

          let colorScale = d3.scaleOrdinal()
            .range(['orange', 'blue', 'green'])
            .domain(roles)

          let colorAccessor = (d) => colorScale(d.classification)

          let markers = svg.append('g')

          let playerG = markers.selectAll('circle.player-marker')
            .data(data)
            .enter().append('circle')
              .attr('class', 'player-marker')
              .attr('id', d => 'marker-' + d.Player_name.replace(/\s/g, ""))
              .attr("cx", function (d) { return projection([+d.Longitude, +d.Latitude])[0] })
              .attr("cy", function (d) { return projection([+d.Longitude, +d.Latitude])[1] })
              .attr('r', 1.5)
              .attr("fill-opacity", .3)
              .attr('fill', colorAccessor)
              .attr('stroke', colorAccessor)
              .attr("stroke-width", 0.5)
              .style('cursor', 'pointer')
              .on("click", function(){
                mouseover
                clicked = !clicked
              })
              .on("mouseover", mouseover)
              .on("mouseleave", mouseleave)

          drawLegend(roles, colorScale, d3.select('svg'))  

        }

        function drawLegend(data, scale, svg) {

          let R = 8
          const svgLegend = svg.append('g')
            .attr('class', 'legend')
            .attr('transform', 'translate(-30, 30)')

          svgLegend.append('text')
            .attr("transform", function (d, i) {return "translate(92," + 25 + ")"})
            .style('font-weight', 'bold')
            .style('font-size', 13)
            .text('Birthplace Population')

          const legend = svgLegend.selectAll('.legend')
            .data(data)
            .enter().append('g')
              .attr("class", "legend")
              .attr("transform", function (d, i) {return "translate(100," + (i+2) * 25 + ")"})

          legend.append("circle")
              .attr("class", "legend-node")
              .attr("cx", 0)
              .attr("cy", 0)
              .attr("r", R)
              .attr("fill", d=>scale(d))

          legend.append("text")
              .attr("class", "legend-text")
              .attr("x", R*2)
              .attr("y", R/2)
              .attr("font-size", '11px')
              .text(d=>d)

        }

        function drawRadiusLegend(size, svg) {

          var valuesToShow = [50, 200, 400]
          var xCircle = 120
          var xLabel = 180
          var yCircle = 245

          const svgLegend = svg.append('g')
            .attr('class', 'radius-legend')
            .attr('transform', 'translate(0, 0)')

          svgLegend.append('text')
            .attr("transform", function (d, i) {return "translate(92," + 190 + ")"})
            .style('font-weight', 'bold')
            .style('font-size', 13)
            .text('Number of ODIs Played')

          svgLegend
            .selectAll(".radius-legend")
            .data(valuesToShow)
            .enter()
            .append("circle")
              .attr("cx", xCircle)
              .attr("cy", function(d){ return yCircle - size(d) } )
              .attr("r", function(d){ return size(d) })
              .style("fill", "none")
              .attr("stroke", "black")

          // Add legend: segments
          svgLegend
            .selectAll(".radius-legend")
            .data(valuesToShow)
            .enter()
            .append("line")
              .attr('x1', function(d){ return xCircle + size(d) } )
              .attr('x2', xLabel)
              .attr('y1', function(d){ return yCircle - size(d) } )
              .attr('y2', function(d){ return yCircle - size(d) } )
              .attr('stroke', 'black')
              .style('stroke-dasharray', ('2,2'))

          // Add legend: labels
          svg
            .selectAll("legend")
            .data(valuesToShow)
            .enter()
            .append("text")
              .attr('x', xLabel)
              .attr('y', function(d){ return yCircle - size(d) } )
              .text( function(d){ return d } )
              .style("font-size", 8)
              .attr('alignment-baseline', 'middle')

        }

        function onlyUnique(value, index, self) { 
            return self.indexOf(value) === index;
        }

        function classification(d){
          if(d > 6000000){
            return '> 6 million'
          } else if(d > 1000000 & d <= 6000000){
            return '1-6 million'
          } else if(d <= 1000000){
            return '< 1 million'
          }
        }

    })
"""