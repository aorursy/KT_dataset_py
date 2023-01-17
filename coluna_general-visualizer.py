import IPython
import datetime
if 'finalGv' in globals():
    IPython.get_ipython().events.unregister('post_execute', finalGv)
gvScriptLines = []
def finalGv():
    if 1<=len(gvScriptLines):
        finalGv.idx += 1
        divId = 'gv' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + ('_%s' % finalGv.idx)
        html1 = '''<div id="%s" style="position: relative; left: 0px; top: 0px; width: 100%%; height: 400px;"></div>''' % divId
        # Generated by Haxe 3.4.7
        js2 = '''(function(t,I){function E(a,c){if(null==c)return null;null==c.__id__&&(c.__id__=J++);var b;null==a.hx__closures__?a.hx__closures__={}:b=a.hx__closures__[c.__id__];null==b&&(b=function(){return b.method.apply(b.scope,arguments)},b.scope=a,b.method=c,a.hx__closures__[c.__id__]=b);return b}var C=function(){};C.__name__=!0;C.strDate=function(a){switch(a.length){case 8:a=a.split(":");var c=new Date;c.setTime(0);c.setUTCHours(a[0]);c.setUTCMinutes(a[1]);c.setUTCSeconds(a[2]);return c;case 10:return a=a.split("-"), new Date(a[0],a[1]-1,a[2],0,0,0);case 19:return c=a.split(" "),a=c[0].split("-"),c=c[1].split(":"),new Date(a[0],a[1]-1,a[2],c[0],c[1],c[2]);default:throw new m("Invalid date format : "+a);}};C.iter=function(a){return{cur:0,arr:a,hasNext:function(){return this.cur<this.arr.length},next:function(){return this.arr[this.cur++]}}};var A=function(){this.length=0};A.__name__=!0;A.prototype={add:function(a){a=new B(a,null);null==this.h?this.h=a:this.q.next=a;this.q=a;this.length++},__class__:A};var B=function(a, c){this.item=a;this.next=c};B.__name__=!0;B.prototype={__class__:B};Math.__name__=!0;var w=function(){};w.__name__=!0;w.string=function(a){return g.__string_rec(a,"")};var f=function(){};f.__name__=!0;f.initGv=t.gv=function(a){f.core=new k;f.main=new u(f.core,a)};f.newTime=t.n=function(a){f.core.newTime(a);f.main.update()};f.circle=t.c=function(a,c,b){null==b&&(b=.5);a=new v(a,c,b);f.core.addItem(a);f.main.update();return a};f.text=t.t=function(a,c,b,d){null==d&&(d=.5);a=new x(a,c,b,d);f.core.addItem(a); f.main.update();return a};f.polygon=t.p=function(){for(var a=arguments,c=new y,b=0,d=g.__cast(a.length/2,F);b<d;){var l=b++;c.add(a[2*l],a[2*l+1])}f.core.addItem(c);f.main.update();return c};f.line=t.l=function(a,c,b,d,l){null==l&&(l=.5);var e=new y,n=b-a,h=d-c;l/=Math.sqrt(n*n+h*h);n*=l;h*=l;e.add(b+.05/(1+Math.sqrt(2))*h,d+.05/(1+Math.sqrt(2))*n);e.add(b-.05*Math.sqrt(2)/(1+Math.sqrt(2))*n-.05*h,d-.05*Math.sqrt(2)/(1+Math.sqrt(2))*h+.05*n);e.add(a+.05*Math.sqrt(2)/(1+Math.sqrt(2))*n-.05*h,c+.05* Math.sqrt(2)/(1+Math.sqrt(2))*h+.05*n);e.add(a-.05/(1+Math.sqrt(2))*h,c+.05/(1+Math.sqrt(2))*n);e.add(a+.05/(1+Math.sqrt(2))*h,c-.05/(1+Math.sqrt(2))*n);e.add(a+.05*Math.sqrt(2)/(1+Math.sqrt(2))*n+.05*h,c+.05*Math.sqrt(2)/(1+Math.sqrt(2))*h-.05*n);e.add(b-.05*Math.sqrt(2)/(1+Math.sqrt(2))*n+.05*h,d-.05*Math.sqrt(2)/(1+Math.sqrt(2))*h-.05*n);e.add(b+.05/(1+Math.sqrt(2))*h,d-.05/(1+Math.sqrt(2))*n);f.core.addItem(e);f.main.update();return e};f.out=t.o=function(a){f.core.addOut(a);f.main.update()};f.autoMode= t.a=function(){f.core.autoMode();f.main.update()};var k=function(){this.nowDragFlag=!1;this.inputInt_=this.inputFloat_=this.dragStartInt_=this.dragStartFloat_=this.dragMoveInt_=this.dragMoveFloat_=this.dragEnd_=null;this.autoModeCount=0;this.outMap=new z;this.snapMap=new z;this.emptyFlag=!0;this.maxX=this.maxY=1;this.nowTime=this.maxTime=this.minX=this.minY=0};k.__name__=!0;k.gvGetColorFromIndex=function(a){return k.colors[a]};k.rgb=function(a,c,b){return"rgb("+100*a+"%, "+100*c+"%, "+100*b+"%)"}; k.prototype={newTime:function(a){null==a?this.nowTime=.1+Math.max(0,this.maxTime+1)|0:(this.maxTime=.1+Math.max(this.maxTime,a)|0,this.nowTime=a)},addItem:function(a){this.emptyFlag?(this.emptyFlag=!1,this.minX=a.getMinX(),this.minY=a.getMinY(),this.maxX=a.getMaxX(),this.maxY=a.getMaxY(),this.maxTime=this.nowTime):(this.minX=Math.min(this.minX,a.getMinX()),this.minY=Math.min(this.minY,a.getMinY()),this.maxX=Math.max(this.maxX,a.getMaxX()),this.maxY=Math.max(this.maxY,a.getMaxY()),this.maxTime=.1+ Math.max(this.maxTime,this.nowTime)|0);if(this.snapMap.h.hasOwnProperty(this.nowTime))this.snapMap.h[this.nowTime].addItem(a);else{var c=new D(this.nowTime);this.snapMap.h[this.nowTime]=c;c.addItem(a)}},addOut:function(a){this.outMap.h.hasOwnProperty(this.nowTime)?this.outMap.h[this.nowTime]=""+this.outMap.h[this.nowTime]+a+"\\n":this.outMap.h[this.nowTime]=""+a+"\\n"},getMinX:function(){return this.minX},getMinY:function(){return this.minY},getMaxX:function(){return this.maxX},getMaxY:function(){return this.maxY}, getTimeList:function(){for(var a=[],c=this.snapMap.keys();c.hasNext();){var b=c.next();a.push(b)}return a},getSnap:function(a){return this.snapMap.h[a]},getOut:function(a){return this.outMap.h.hasOwnProperty(a)?this.outMap.h[a]:""},getAutoModeCount:function(){return this.autoModeCount},sendInput:function(a,c,b){if(null!=this.inputInt_){var d=this.inputInt_;this.inputInt_=null;d(a,Math.round(c),Math.round(b))}else null!=this.inputFloat_&&(d=this.inputFloat_,this.inputFloat_=null,d(a,c,b))},inputInt:function(a){this.inputInt_= a;this.inputFloat_=null},inputFloat:function(a){this.inputFloat_=a;this.inputInt_=null},setDragModeInt:function(a,c,b){this.dragStartInt_=a;this.dragStartFloat_=null;this.dragMoveInt_=c;this.dragMoveFloat_=null;this.dragEnd_=b},setDragModeFloat:function(a,c,b){this.dragStartInt_=a;this.dragStartFloat_=null;this.dragMoveInt_=c;this.dragMoveFloat_=null;this.dragEnd_=b},isDragMode:function(){return null==this.dragStartInt_?null!=this.dragStartFloat_:!0},isNowDrag:function(){return this.nowDragFlag}, sendDragStart:function(a,c,b){this.sendDragEnd();if(null!=this.dragStartInt_){var d=this.dragStartInt_;this.nowDragFlag=!0;d(a,Math.round(c),Math.round(b))}else null!=this.dragStartFloat_&&(d=this.dragStartFloat_,this.nowDragFlag=!0,d(a,c,b))},sendDragMove:function(a,c,b){this.nowDragFlag&&(null!=this.dragMoveInt_?(a=this.dragMoveInt_,a(Math.round(c),Math.round(b))):null!=this.dragMoveFloat_&&(a=this.dragMoveFloat_,a(c,b)))},sendDragEnd:function(){if(this.nowDragFlag){if(null!=this.dragEnd_){var a= this.dragEnd_;a()}this.nowDragFlag=!1}},autoMode:function(){++this.autoModeCount},__class__:k};var u=function(a,c){this.updateTimer=this.paintTimer=this.autoModeTimerId=null;this.autoModeCount=0;this.autoMode=!1;this.cx=this.cy=this.mx=this.my=this.cursorX=this.cursorY=this.myMouseX=this.myMouseY=0;this.scale=1;this.nowSnap=null;this.now=0;this.core=this.timeList=null;var b=this;this.core=a;this.parent=window.document.getElementById(c);this.canvas=window.document.createElement("canvas");this.parent.appendChild(this.canvas); this.canvas.style.position="absolute";this.canvas.style.left="0px";this.canvas.style.top="0px";this.canvas.style.width="100%";this.canvas.style.height="100%";this.canvas.width=this.canvas.clientWidth;this.canvas.height=this.canvas.clientHeight;this.canvas.tabIndex=-1;this.ctx=this.canvas.getContext("2d",null);this.div=window.document.createElement("pre");this.parent.appendChild(this.div);this.div.style.position="absolute";this.div.style.left="0px";this.div.style.bottom="0px";this.div.style.width= "100%";this.div.style.overflow="hidden";var d=window.document.createElement("button");this.parent.appendChild(d);d.style.position="absolute";d.style.right="0px";d.style.bottom="0px";d.textContent="full";var l=!1,e=this.parent.style.position,n=this.parent.style.left,h=this.parent.style.top,g=this.parent.style.width,f=this.parent.style.height,p=this.parent.style.zIndex,k=null;d.onclick=function(a){if(l=!l){for(a=b.parent.parentElement;null!=a;){if(a.classList.contains("output_wrapper")){k=a.style.zIndex; a.style.zIndex="auto";break}a=a.parentElement}b.parent.style.position="fixed";b.parent.style.left="0px";b.parent.style.top="0px";b.parent.style.width="100%";b.parent.style.height="100%";b.parent.style.zIndex="100";d.textContent="esc";window.onresize=function(a){b.updateUI()}}else{for(a=b.parent.parentElement;null!=a;){if(a.classList.contains("output_wrapper")){a.style.zIndex=k;break}a=a.parentElement}b.parent.style.position=e;b.parent.style.left=n;b.parent.style.top=h;b.parent.style.width=g;b.parent.style.height= f;b.parent.style.zIndex=p;d.textContent="full";window.onresize=null}b.updateUI();return!1};this.canvas.onkeydown=function(a){switch(a.keyCode){case 33:b.autoMode=!1;1<=b.now&&(b.now=Math.max(0,b.now-(Math.max(10,Math.sqrt(b.timeList.length))|0))|0,b.updateTime());break;case 34:b.autoMode=!1;null!=b.timeList&&b.now<b.timeList.length-1&&(b.now=Math.min(b.now+Math.max(10,Math.sqrt(b.timeList.length)|0),b.timeList.length-1)|0,b.updateTime());break;case 35:b.autoMode=!1;null!=b.timeList&&b.now<b.timeList.length- 1&&(b.now=b.timeList.length-1,b.updateTime());break;case 36:b.autoMode=!1;1<=b.now&&(b.now=0,b.updateTime());break;case 37:b.autoMode=!1;1<=b.now&&(--b.now,b.updateTime());break;case 38:b.updateSelf(null,!1,4,!1,!1);break;case 39:b.autoMode=!1;null!=b.timeList&&b.now<b.timeList.length-1&&(++b.now,b.updateTime());break;case 40:b.updateSelf(null,!1,-4,!1,!1);break;case 97:b.onNumpadKey(-.7,.7);break;case 98:b.onNumpadKey(0,1);break;case 99:b.onNumpadKey(.7,.7);break;case 100:b.onNumpadKey(-1,0);break; case 102:b.onNumpadKey(1,0);break;case 103:b.onNumpadKey(-.7,-.7);break;case 104:b.onNumpadKey(0,-1);break;case 105:b.onNumpadKey(.7,-.7)}};var m=!1;this.canvas.onmousedown=function(a){b.canvas.focus();m=!0;b.myMouseX=a.clientX;b.myMouseY=a.clientY;b.updateSelf(null,!1,0,!1,a.shiftKey);return!1};this.canvas.onmouseup=function(c){m=!1;b.myMouseX=c.clientX;b.myMouseY=c.clientY;b.updateSelf(null,!1,0,!1,!1);a.isDragMode()&&a.isNowDrag()&&(a.sendDragEnd(),b.updateTimeList());return!1};this.canvas.onmousemove= function(a){b.myMouseX=a.clientX;b.myMouseY=a.clientY;b.updateSelf(null,m,0,!1,!1);return!1};this.canvas.onwheel=function(a){b.myMouseX=a.clientX;b.myMouseY=a.clientY;var c=0<a.detail?-1:0>a.detail?1:0;0==c&&(a=a.wheelDelta,c=0<a?1:0>a?-1:0);b.updateSelf(null,!1,c,!1,!1);return!1};var r=null,q=null,t=0,u=0,y=12.425134878021495/Math.log(2),v=new z,x=function(a){u!=a.touches.length&&(r=null);u=a.touches.length;if(1<=a.touches.length){for(var c=0,d=a.touches.length;c<d;){var e=c++;e=a.touches.item(e); v.h.hasOwnProperty(e.identifier)||(r=null)}e=d=c=0;for(var l=a.touches.length;e<l;){var h=e++;h=a.touches.item(h);c+=h.pageX;d+=h.pageY}c/=a.touches.length;d/=a.touches.length;l=e=0;for(h=a.touches.length;l<h;){var g=l++,f=a.touches.item(g);g=f.pageX-c;f=f.pageY-d;e+=Math.sqrt(g*g+f*f+1E-5)}e/=a.touches.length;null!=r?3<=a.touches.length?(b.autoMode=!1,c=10*(c-r)/b.canvas.width,c=t-(0<=c?Math.floor(c):Math.ceil(c)),c!=b.now&&null!=b.timeList&&0<=c&&c<b.timeList.length&&(b.now=c,b.updateTime())):2== a.touches.length?(l=Math.log(e/q)*y,b.myMouseX=c,b.myMouseY=d,b.updateSelf(null,!1,l,!1,!1),r=c,q=e):1==a.touches.length&&(b.myMouseX=c,b.myMouseY=d,b.updateSelf(null,!0,0,!1,!1),r=c,q=e):(b.myMouseX=c,b.myMouseY=d,b.updateSelf(null,!1,0,!1,!1),r=c,q=e,t=b.now)}v=new z;c=0;for(d=a.touches.length;c<d;)e=c++,e=a.touches.item(e),v.h[e.identifier]=!0;a.preventDefault();return!1};this.canvas.ontouchmove=x;var A=null,B=null,w=null,D=C.strDate("2000-01-01 00:00:01").getTime()-C.strDate("2000-01-01 00:00:00").getTime(); this.canvas.ontouchstart=function(a){r=null;if(1==a.touches.length){var c=a.touches.item(0).pageX,d=a.touches.item(0).pageY,e=(new Date).getTime();if(null!=w&&e-w<=.5*D){var l=c-A,h=d-B;if(Math.sqrt(l*l+h*h)<=.05*Math.min(b.canvas.width,b.canvas.height))return b.myMouseX=c,b.myMouseY=d,b.updateSelf(null,!1,0,!1,!0),a.preventDefault(),!1}A=c;B=d;w=e}x(a);return!1};this.canvas.ontouchcancel=function(a){r=null;a.preventDefault();return!1};this.canvas.ontouchend=function(c){r=null;c.preventDefault(); c.targetTouches.length==c.touches.length&&a.isDragMode()&&a.isNowDrag()&&(a.sendDragEnd(),b.updateTimeList());return!1};this.update()};u.__name__=!0;u.main=function(){};u.prototype={update:function(){var a=this;null==this.updateTimer&&(this.updateTimer=window.setTimeout(function(){a.updateTimer=null;a.updateTimeList()},10))},onNumpadKey:function(a,c){var b=Math.min(Math.max(-this.mx,this.cx+a*this.scale*.25),this.mx),d=Math.min(Math.max(-this.my,this.cy+c*this.scale*.25),this.my);if(this.cx!=b||this.cy!= d)this.cx=b,this.cy=d,this.updateUI()},updateUI:function(){null!=this.paintTimer&&window.clearTimeout(this.paintTimer);this.paintTimer=window.setTimeout(E(this,this.paintSelf),10)},paintSelf:function(){this.paintTimer=null;if(this.canvas.width!=this.canvas.clientWidth||this.canvas.height!=this.canvas.clientHeight)this.canvas.width=this.canvas.clientWidth,this.canvas.height=this.canvas.clientHeight;this.updateSelf(this.ctx,!1,0,!1,!1)},updateSelf:function(a,c,b,d,l){var e=Math.max(1,this.canvas.width), f=Math.max(1,this.canvas.height),h=this.core.getMaxX()-this.core.getMinX(),g=this.core.getMaxY()-this.core.getMinY(),k=Math.max(h,g);if(h*f<g*e){this.my=.5*(1-this.scale);var p=f/(g*this.scale);this.mx=p*h<=e?0:(h-e/p)/k*.5}else this.mx=.5*(1-this.scale),p=e/(h*this.scale),this.my=p*g<=f?0:(g-f/p)/k*.5;this.updateCenter();var q=this.cursorX,m=this.cursorY;if(d)this.cx=(this.cursorX-(this.myMouseX-.5*e)/p-.5*h-this.core.getMinX())/k,this.cy=(this.cursorY-(this.myMouseY-.5*f)/p-.5*g-this.core.getMinY())/ k,this.updateCenter();else if(this.cursorX=(this.myMouseX-.5*e)/p+.5*h+this.core.getMinX()+k*this.cx,this.cursorY=(this.myMouseY-.5*f)/p+.5*g+this.core.getMinY()+k*this.cy,null!=this.nowSnap){d=this.nowSnap.getTime();if(c)if(this.core.isDragMode())this.core.isNowDrag()||this.core.sendDragStart(d,q,m),this.core.sendDragMove(d,this.cursorX,this.cursorY),this.updateTimeList();else{c=this.cursorY-m;m=this.cx;var r=this.cy;this.cx-=(this.cursorX-q)/k;this.cy-=c/k;this.updateCenter();if(m!=this.cx||r!= this.cy){0!=b&&(this.cursorX=(this.myMouseX-.5*e)/p+.5*h+this.core.getMinX()+k*this.cx,this.cursorY=(this.myMouseY-.5*f)/p+.5*g+this.core.getMinY()+k*this.cy,a=Math.min(Math.max(.01,this.scale*Math.pow(.5,.080482023721841*b)),1),this.scale!=a&&(this.scale=a,this.updateSelf(null,!1,0,!0,!1)));this.updateUI();return}}if(0!=b&&(b=Math.min(Math.max(.01,this.scale*Math.pow(.5,.080482023721841*b)),1),this.scale!=b)){this.scale=b;this.updateSelf(null,!1,0,!0,!1);this.updateUI();return}l&&(this.core.sendInput(d, this.cursorX,this.cursorY),this.updateTimeList());l=this.core.getOut(d);0<=this.myMouseX&&0<=this.myMouseY&&this.core.getMinX()<=this.cursorX&&this.cursorX<=this.core.getMaxX()&&this.core.getMinY()<=this.cursorY&&this.cursorY<=this.core.getMaxY()?this.div.textContent=""+l+"time "+d+" ( "+(this.now+1)+" / "+this.timeList.length+" ) ("+(this.cursorX+.5|0)+", "+(this.cursorY+.5|0)+") ("+this.cursorX+", "+this.cursorY+")":this.div.textContent=""+l+"time "+d+" ( "+(this.now+1)+" / "+this.timeList.length+ " )";h=.5*(e/p-h)-this.core.getMinX()-k*this.cx;g=.5*(f/p-g)-this.core.getMinY()-k*this.cy;null!=a&&(a.clearRect(0,0,e,f),a.save(),a.scale(p,p),a.translate(h,g),this.nowSnap.paint(a),a.restore())}},updateCenter:function(){this.cx=Math.min(Math.max(-this.mx,this.cx),this.mx);this.cy=Math.min(Math.max(-this.my,this.cy),this.my)},setAutoModeTimer:function(){null!=this.autoModeTimerId&&window.clearTimeout(this.autoModeTimerId);this.autoModeTimerId=window.setTimeout(E(this,this.onAutoModeTimer),200)}, onAutoModeTimer:function(){null!=this.autoModeTimerId&&(window.clearTimeout(this.autoModeTimerId),this.autoModeTimerId=null);null!=this.timeList&&this.now<this.timeList.length-1&&(++this.now,this.updateTime(),this.setAutoModeTimer())},updateTime:function(){if(null!=this.timeList&&this.now<this.timeList.length){var a=this.timeList[this.now];this.now==this.timeList.length-1&&(this.autoMode=!0);this.nowSnap=this.core.getSnap(a);this.nowSnap.output();this.updateUI();a=this.core.getAutoModeCount();a!= this.autoModeCount&&(this.autoModeCount=a,this.autoMode=!0);this.autoMode&&this.setAutoModeTimer()}else this.nowSnap=null},updateTimeList:function(){var a=null!=this.timeList&&this.now<this.timeList.length?this.timeList[this.now]:0;this.timeList=this.core.getTimeList();if(null!=this.timeList&&0<this.timeList.length){var c=Math.abs(a-this.timeList[0]);this.now=0;for(var b=1,d=this.timeList.length;b<d;){var g=b++,e=Math.abs(a-this.timeList[g]);e<c&&(c=e,this.now=g)}this.updateTime()}},__class__:u}; var D=function(a){this.time=a;this.items=new A};D.__name__=!0;D.prototype={addItem:function(a){this.items.add(a)},paint:function(a){for(var c=this.items.h;null!=c;){var b=c.item;c=c.next;b.paint(a)}},output:function(){for(var a=this.items.h;null!=a;){var c=a.item;a=a.next;c.output()}},getTime:function(){return this.time},__class__:D};var q=function(){};q.__name__=!0;q.prototype={__class__:q};var v=function(a,c,b){this.x=a;this.y=c;this.r=b;this.colorB=this.colorG=this.colorR=0};v.__name__=!0;v.__interfaces__= [q];v.prototype={rgb:function(a,c,b){this.colorR=a/255;this.colorG=c/255;this.colorB=b/255;return this},color:function(a){a=k.gvGetColorFromIndex(a);this.colorR=a[0];this.colorG=a[1];this.colorB=a[2];return this},getMinX:function(){return this.x-this.r},getMinY:function(){return this.y-this.r},getMaxX:function(){return this.x+this.r},getMaxY:function(){return this.y+this.r},paint:function(a){a.fillStyle=k.rgb(this.colorR,this.colorG,this.colorB);a.beginPath();a.arc(this.x,this.y,this.r,0,2*Math.PI, !1);a.fill()},output:function(){},__class__:v};var y=function(){this.yVec=[];this.xVec=[];this.colorB=this.colorG=this.colorR=0};y.__name__=!0;y.__interfaces__=[q];y.prototype={rgb:function(a,c,b){this.colorR=a/255;this.colorG=c/255;this.colorB=b/255;return this},color:function(a){a=k.gvGetColorFromIndex(a);this.colorR=a[0];this.colorG=a[1];this.colorB=a[2];return this},add:function(a,c){this.xVec.push(a);this.yVec.push(c);return this},getMinX:function(){for(var a=Infinity,c=0,b=this.xVec;c<b.length;){var d= b[c];++c;a=Math.min(a,d)}return a},getMinY:function(){for(var a=Infinity,c=0,b=this.yVec;c<b.length;){var d=b[c];++c;a=Math.min(a,d)}return a},getMaxX:function(){for(var a=-Infinity,c=0,b=this.xVec;c<b.length;){var d=b[c];++c;a=Math.max(a,d)}return a},getMaxY:function(){for(var a=-Infinity,c=0,b=this.yVec;c<b.length;){var d=b[c];++c;a=Math.max(a,d)}return a},paint:function(a){var c=this.xVec.length;if(0<c){a.fillStyle=k.rgb(this.colorR,this.colorG,this.colorB);a.beginPath();a.moveTo(this.xVec[c-1], this.yVec[c-1]);for(var b=0;b<c;){var d=b++;a.lineTo(this.xVec[d],this.yVec[d])}a.fill()}},output:function(){},__class__:y};var x=function(a,c,b,d){this.x=c;this.y=b;this.r=d;this.text=a;this.colorB=this.colorG=this.colorR=0};x.__name__=!0;x.__interfaces__=[q];x.prototype={rgb:function(a,c,b){this.colorR=a/255;this.colorG=c/255;this.colorB=b/255;return this},color:function(a){a=k.gvGetColorFromIndex(a);this.colorR=a[0];this.colorG=a[1];this.colorB=a[2];return this},getMinX:function(){return this.x- this.r},getMinY:function(){return this.y-this.r},getMaxX:function(){return this.x+this.r},getMaxY:function(){return this.y+this.r},paint:function(a){var c=.02*this.r;a.save();a.translate(this.x,this.y);a.font="100px hoge";a.scale(c,c);a.fillStyle=k.rgb(this.colorR,this.colorG,this.colorB);a.textAlign="center";a.textBaseline="middle";a.fillText(this.text,0,0);a.restore()},output:function(){},__class__:x};q=function(){};q.__name__=!0;var z=function(){this.h={}};z.__name__=!0;z.__interfaces__=[q];z.prototype= {keys:function(){var a=[],c;for(c in this.h)this.h.hasOwnProperty(c)&&a.push(c|0);return C.iter(a)},__class__:z};var m=function(a){Error.call(this);this.val=a;this.message=String(a);Error.captureStackTrace&&Error.captureStackTrace(this,m)};m.__name__=!0;m.wrap=function(a){return a instanceof Error?a:new m(a)};m.__super__=Error;m.prototype=function(a,c){function b(){}b.prototype=a;var d=new b,g;for(g in c)d[g]=c[g];c.toString!==Object.prototype.toString&&(d.toString=c.toString);return d}(Error.prototype, {__class__:m});var g=function(){};g.__name__=!0;g.getClass=function(a){if(a instanceof Array&&null==a.__enum__)return Array;var c=a.__class__;if(null!=c)return c;a=g.__nativeClassName(a);return null!=a?g.__resolveNativeClass(a):null};g.__string_rec=function(a,c){if(null==a)return"null";if(5<=c.length)return"<...>";var b=typeof a;"function"==b&&(a.__name__||a.__ename__)&&(b="object");switch(b){case "function":return"<function>";case "object":if(a instanceof Array){if(a.__enum__){if(2==a.length)return a[0]; b=a[0]+"(";c+="\\t";for(var d=2,f=a.length;d<f;){var e=d++;b=2!=e?b+(","+g.__string_rec(a[e],c)):b+g.__string_rec(a[e],c)}return b+")"}b=a.length;d="[";c+="\\t";for(f=0;f<b;)e=f++,d+=(0<e?",":"")+g.__string_rec(a[e],c);return d+"]"}try{d=a.toString}catch(n){return"???"}if(null!=d&&d!=Object.toString&&"function"==typeof d&&(b=a.toString(),"[object Object]"!=b))return b;b=null;d="{\\n";c+="\\t";f=null!=a.hasOwnProperty;for(b in a)f&&!a.hasOwnProperty(b)||"prototype"==b||"__class__"==b||"__super__"==b|| "__interfaces__"==b||"__properties__"==b||(2!=d.length&&(d+=", \\n"),d+=c+b+" : "+g.__string_rec(a[b],c));c=c.substring(1);return d+("\\n"+c+"}");case "string":return a;default:return String(a)}};g.__interfLoop=function(a,c){if(null==a)return!1;if(a==c)return!0;var b=a.__interfaces__;if(null!=b)for(var d=0,f=b.length;d<f;){var e=d++;e=b[e];if(e==c||g.__interfLoop(e,c))return!0}return g.__interfLoop(a.__super__,c)};g.__instanceof=function(a,c){if(null==c)return!1;switch(c){case Array:return a instanceof Array?null==a.__enum__:!1;case G:return"boolean"==typeof a;case K:return!0;case H:return"number"==typeof a;case F:return"number"==typeof a?(a|0)===a:!1;case String:return"string"==typeof a;default:if(null!=a)if("function"==typeof c){if(a instanceof c||g.__interfLoop(g.getClass(a),c))return!0}else{if("object"==typeof c&&g.__isNativeObj(c)&&a instanceof c)return!0}else return!1;return c==L&&null!=a.__name__||c==M&&null!=a.__ename__?!0:a.__enum__==c}};g.__cast=function(a,c){if(g.__instanceof(a,c))return a; throw new m("Cannot cast "+w.string(a)+" to "+w.string(c));};g.__nativeClassName=function(a){a=g.__toStr.call(a).slice(8,-1);return"Object"==a||"Function"==a||"Math"==a||"JSON"==a?null:a};g.__isNativeObj=function(a){return null!=g.__nativeClassName(a)};g.__resolveNativeClass=function(a){return I[a]};var J=0;String.prototype.__class__=String;String.__name__=!0;Array.__name__=!0;Date.prototype.__class__=Date;Date.__name__=["Date"];var F={__name__:["Int"]},K={__name__:["Dynamic"]},H=Number;H.__name__= ["Float"];var G=Boolean;G.__ename__=["Bool"];var L={__name__:["Class"]},M={};k.colors=[[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,.5,0],[1,0,.5]];g.__toStr={}.toString;u.main()})("undefined"!=typeof exports?exports:"undefined"!=typeof window?window:"undefined"!=typeof self?self:this,"undefined"!=typeof window?window:"undefined"!=typeof global?global:"undefined"!=typeof self?self:this);'''
        IPython.display.display(IPython.display.HTML('''%s<script>%sgv("%s");%s</script>''' % (html1, js2, divId, ';'.join(gvScriptLines))))
        gvScriptLines.clear()
IPython.get_ipython().events.register('post_execute', finalGv)
finalGv.idx = 0
def color(r, g, b):
    finalGv.default_color = (r, g, b)
finalGv.default_color = (0, 0, 0)
def circle(x, y, w=1.0, color=None):
    if color is None:
        color = finalGv.default_color
    gvScriptLines.append('''c(%s, %s, %s).rgb(%s, %s, %s)''' % (x, y, w*0.5, color[0], color[1], color[2]))
def rect(x, y, w=1.0, color=None):
    if color is None:
        color = finalGv.default_color
    gvScriptLines.append('''p(%s, %s, %s, %s, %s, %s, %s, %s).rgb(%s, %s, %s)''' % (x-w*0.5, y-w*0.5, x+w*0.5, y-w*0.5, x+w*0.5, y+w*0.5, x-w*0.5, y+w*0.5, color[0], color[1], color[2]))
def text(x, y, t='?', w=1.0, color=None):
    if color is None:
        color = finalGv.default_color
    gvScriptLines.append('''t("%s", %s, %s, %s).rgb(%s, %s, %s)''' % (t.replace('"', '\\"'), x, y, w*0.5, color[0], color[1], color[2]))
def line(sx, sy, ex, ey, w=1.0, color=None):
    if color is None:
        color = finalGv.default_color
    gvScriptLines.append('''l(%s, %s, %s, %s, %s).rgb(%s, %s, %s)''' % (sx, sy, ex, ey, w*0.5, color[0], color[1], color[2]))
def newTime():
    gvScriptLines.append('''n()''')

color(255, 0, 0)
circle(1, 1)
color(0, 255, 0)
circle(2, 1)
color(0, 0, 255)
circle(1, 2)
color(255, 128, 0)
rect(2, 2)
newTime()
circle(1, 1, color=(255, 0, 0))
circle(2, 1, color=(0, 255, 0))
circle(1, 2, color=(0, 0, 255))
rect(2, 2, color=(255, 128, 0))
line(1, 1, 2, 2, color=(0, 0, 0))
text(1, 1)
text(2, 2, 'A', color=(0, 255, 255))