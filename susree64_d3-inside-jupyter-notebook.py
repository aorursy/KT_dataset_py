# Embedding d3.js inside html and running in Jupyter Notebook
# Import HTML library

from IPython.core.display import display, HTML
HTML('''

<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="utf-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <script src="http://d3js.org/d3.v3.min.js" charset="utf-8"></script>

    <title>Document</title>

    <!-- This is css block-->

 

     <style>

    </style>

</head>

<body>

   <h1 id = 'caption'>Earthquakes Bubble Graph</h1>

   <div><br></br></div>

   <div class = 'canvas'></div>



<!--Script for d3 js effects and data manipulation -->   

 <script  type="text/javascript">

   d3.select('#caption')

  .style('color','red')

  .style('text-align', 'center')

  .style('font-size', '32px')

  canv = d3.select('.canvas');

  url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson";

  var width = 1000 ; var height = 400;

// random numbers genertor function betwee a range

function between(min, max) {  

    return Math.floor(

      Math.random() * (max - min) + min

    )

  }

  

  svg = canv.append('svg');

  svg.attr('width', width);

  svg.attr('height', height);

  svg.style('border',"5px solid green");

  

  // centering the svg block ;

  svg.style('background-color', '#a4c2da');

  svg.style('margin-left', 'auto');

  svg.style('margin-right','auto');

  svg.style('display', 'block');

  

  value = svg.append('text');

    value.text('') //function(d,i,n) {return(d.properties.mag);

    value.attr('x', 250)

    value.attr('y', 30)

    value.attr('font-size', '32px')

 

 d3.json(url, function(error, data){

        console.log(data)

        circle = svg.selectAll('circle');

        circle.data(data.features)

        .enter().append('circle')

        // circles cx parameter is taken from the magnitudue of earthquakes

         .attr('cx', function (d,i,n) {return (between(30,850) + d.properties.mag*i);})

         .attr('cy', function (d,i,n) {return (between(100, 300) + d.properties.mag)})

         .attr('r', function (d, i, n){return (d.properties.mag*6)})

         .attr('fill', function(d,i){return (d.properties.alert)})

         .style('stroke', 'black')

         .style('stroke-width',"3")

         // Mouse over the circles change the opacity

         .on("mouseover", function (d){

                d3.select(this).transition()

                .duration(100)

                .style('opacity', 0.5)

                value.text(d.properties.place + " - " + d.properties.mag) })

      // Mouse out the circles returning back 

         .on("mouseout", function (){

                d3.select(this)

                .transition()

                .duration(100)

                .style('opacity', 1)

                 value.text('')

         })

 })

    

   </script>

</body>

</html>''')